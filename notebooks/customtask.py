# %% [markdown]
# <img src="../assets/CoLLIE_blue.png" alt="GoLLIE" width="200"/>

# %% [markdown]
# # Custom Tasks with GoLLIE
# 
# This notebook is an example of how to run Custom Tasks with GoLLIE. This notebook covers:
# 
# - How to define the guidelines for a task
# - How to load GoLLIE
# - How to generate model inputs
# - How to parse the output
# - How to implement a scorer and evaluate the output
# 
# You can modify this notebook to run any task task you want 

# %% [markdown]
# ### Import requeriments
# 
# See the requeriments.txt file in the main directory to install the required dependencies

# %%
import sys
sys.path.append("../") # Add the GoLLIE base directory to sys path

# %%
import rich
import logging
from src.model.load_model import load_model
import black
import inspect
from jinja2 import Template as jinja2Template
import tempfile
from src.tasks.utils_typing import AnnotationList
logging.basicConfig(level=logging.INFO)
from typing import Dict, List, Type
import torch

# %% [markdown]
# ## Load GoLLIE
# 
# We will load GOLLIE-7B from the huggingface-hub.
# You can use the function AutoModelForCausalLM.from_pretrained if you prefer it. However, we provide a handy load_model function with many functionalities already implemented that will assist you in reproducing our results.
# 
# Please note that setting use_flash_attention=True is mandatory. Our flash attention implementation has small numerical differences compared to the attention implementation in Huggingface. Using use_flash_attention=False will result in the model producing inferior results. Flash attention requires an available CUDA GPU. Running GOLLIE pre-trained models on a CPU is not supported. We plan to address this in future releases.
# 
# - Set force_auto_device_map=True to automatically load the model on available GPUs.
# - Set quantization=4 if the model doesn't fit in your GPU memory.

# %%
model, tokenizer = load_model(
    inference=True,
    model_weights_name_or_path="HiTZ/GoLLIE-7B",
    quantization=None,
    use_lora=False,
    force_auto_device_map=True,
    use_flash_attention=True,
    torch_dtype="bfloat16"
)

# %% [markdown]
# ## Define the guidelines
# 
# First, we will define the labels and guidelines for the task. We will represent them as Python classes.
# 
# The following guidelines have been defined for this example. They were not part of the pre-training dataset. Therefore, we will run GOLLIE in zero-shot settings using unseen labels.
# 
# We will use the `Generic` class, which is a versatile class that allows for the implementation of any task you want. However, since the model has never seen the Generic label during training, we will rename it to Template, which is recognized by the model (as it was used in the Tacred dataset).
# 
# We will define two classes: `Launcher` and `Mission`. Each class will have a definition and a set of slots that the model needs to fill. Each slot also requires a type definition and a short description, which can include examples. For instance, for the `Launcher` class, we define three slots:
# 
# - The `mention`, which will be the name of the Launcher vehicle and should be a string.
# - The `space_company` that operated the vehicle, which will also be a string.
# - The `crew`, which is defined as a list of astronauts. Therefore, GoLLIE will fill this slot with a list of strings.
# 
# 💡 Be creative and try to define your own guidelines to test GoLLIE!

# %%
from typing import List

from src.tasks.utils_typing import dataclass
from src.tasks.utils_typing import Generic as Template

"""
Entity definitions
"""


@dataclass
class Launcher(Template):
    """Refers to a vehicle designed primarily to transport payloads from the Earth's 
    surface to space. Launchers can carry various payloads, including satellites, 
    crewed spacecraft, and cargo, into various orbits or even beyond Earth's orbit. 
    They are usually multi-stage vehicles that use rocket engines for propulsion."""

    mention: str  
    """
    The name of the launcher vehicle. 
    Such as: "Sturn V", "Atlas V", "Soyuz", "Ariane 5"
    """
    space_company: str # The company that operates the launcher. Such as: "Blue origin", "ESA", "Boeing", "ISRO", "Northrop Grumman", "Arianespace"
    crew: List[str] # Names of the crew members boarding the Launcher. Such as: "Neil Armstrong", "Michael Collins", "Buzz Aldrin"
    

@dataclass
class Mission(Template):
    """Any planned or accomplished journey beyond Earth's atmosphere with specific objectives, 
    either crewed or uncrewed. It includes missions to satellites, the International 
    Space Station (ISS), other celestial bodies, and deep space."""
    
    mention: str
    """
    The name of the mission. 
    Such as: "Apollo 11", "Artemis", "Mercury"
    """
    date: str # The start date of the mission
    departure: str # The place from which the vehicle will be launched. Such as: "Florida", "Houston", "French Guiana"
    destination: str # The place or planet to which the launcher will be sent. Such as "Moon", "low-orbit", "Saturn"


ENTITY_DEFINITIONS: List[Template] = [
    Launcher,
    Mission,
]
    
cell_txt = '''
"""
Entity definitions
"""


@dataclass
class Launcher(Template):
    """Refers to a vehicle designed primarily to transport payloads from the Earth's 
    surface to space. Launchers can carry various payloads, including satellites, 
    crewed spacecraft, and cargo, into various orbits or even beyond Earth's orbit. 
    They are usually multi-stage vehicles that use rocket engines for propulsion."""

    mention: str  
    """
    The name of the launcher vehicle. 
    Such as: "Sturn V", "Atlas V", "Soyuz", "Ariane 5"
    """
    space_company: str # The company that operates the launcher. Such as: "Blue origin", "ESA", "Boeing", "ISRO", "Northrop Grumman", "Arianespace"
    crew: List[str] # Names of the crew members boarding the Launcher. Such as: "Neil Armstrong", "Michael Collins", "Buzz Aldrin"
    

@dataclass
class Mission(Template):
    """Any planned or accomplished journey beyond Earth's atmosphere with specific objectives, 
    either crewed or uncrewed. It includes missions to satellites, the International 
    Space Station (ISS), other celestial bodies, and deep space."""
    
    mention: str
    """
    The name of the mission. 
    Such as: "Apollo 11", "Artemis", "Mercury"
    """
    date: str # The start date of the mission
    departure: str # The place from which the vehicle will be launched. Such as: "Florida", "Houston", "French Guiana"
    destination: str # The place or planet to which the launcher will be sent. Such as "Moon", "low-orbit", "Saturn"


ENTITY_DEFINITIONS: List[Template] = [
    Launcher,
    Mission,
]'''


# %% [markdown]
# ### Print the guidelines to guidelines.py
# 
# Due to IPython limitations, we must write the content of the previous cell to a file and then import the content from that file.

# %%
with open("guidelines.py","w",encoding="utf8") as python_guidelines:
    print(cell_txt,file=python_guidelines)

from guidelines import *

# %% [markdown]
# We use inspect.getsource to get the guidelines as a string

# %%
guidelines = [inspect.getsource(definition) for definition in ENTITY_DEFINITIONS]
guidelines

# %% [markdown]
# ## Define input sentence
# 
# Here we define the input sentence and the gold labels.
# 
# You can define and empy list as gold labels if you don't have gold annotations.

# %%
text = "The Ares 3 mission to Mars is scheduled for 2032. The Starship rocket build by SpaceX will take off from Boca Chica, carrying the astronauts Max Rutherford, Elena Soto, and Jake Martinez."
gold = [
    Launcher(mention="Starship",space_company="SpaceX",crew=["Max Rutherford","Elena Soto","Jake Martinez"]),
    Mission(mention="Ares 3",date="2032",departure="Boca Chica",destination="Mars")
]

# %% [markdown]
# ## Filling a template
# 
# We need to define a template. For this task, we will include only the class definitions and the text to be annotated. However, you can design different templates to incorporate more information (for example, event triggers, as demonstrated in the Event Extraction notebook).
# 
# We will use Jinja templates, which are easy to implement and exceptionally fast. For more information, visit: https://jinja.palletsprojects.com/en/3.1.x/api/#high-level-api.
# 
# 

# %%
template_txt =(
"""# The following lines describe the task definition
{%- for definition in guidelines %}
{{ definition }}
{%- endfor %}

# This is the text to analyze
text = {{ text.__repr__() }}

# The annotation instances that take place in the text above are listed here
result = [
{%- for ann in annotations %}
    {{ ann }},
{%- endfor %}
]
""")


# %%

template = jinja2Template(template_txt)
# Fill the template
formated_text = template.render(guidelines=guidelines, text=text, annotations=gold, gold=gold)

# %% [markdown]
# ### Black Code Formatter
# 
# We use the Black Code Formatter to automatically unify all the prompts to the same format. 
# 
# https://github.com/psf/black

# %%
black_mode = black.Mode()
formated_text = black.format_str(formated_text, mode=black_mode)

# %% [markdown]
# ### Print the filled and formatted template

# %%
rich.print(formated_text)

# %% [markdown]
# ## Prepare model inputs
# 
# We remove everything after `result =` to run inference with the model.

# %%
prompt, _ = formated_text.split("result =")
prompt = prompt + "result ="

# %% [markdown]
# Tokenize the input sentence

# %%
model_input = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

# %% [markdown]
# Remove the `eos` token from the input

# %%
model_input["input_ids"] = model_input["input_ids"][:, :-1]
model_input["attention_mask"] = model_input["attention_mask"][:, :-1]

# %% [markdown]
# ## Run GoLLIE
# 
# We generate the predictions using GoLLIE.
# 
# We use `num_beams=1` and `do_sample=False` in our exmperiments. But feel free to experiment with differen decoding strategies 😊

# %%
import torch
print("CUDA Available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))


# %%
import torch

# Ensure a tensor is created on the GPU
tensor = torch.rand(3, 3).cuda()
print("Tensor on GPU:", tensor)


# %%


# %%

import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(3))  # Check if device 3 is available
device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_ouput = model.generate(
    **model_input.to(model.device),
    max_new_tokens=128,
    do_sample=False,
    min_new_tokens=0,
    num_beams=1,
    num_return_sequences=1,
)


# %% [markdown]
# ### Print the results

# %%
for y, x in enumerate(model_ouput):
    print(f"Answer {y}")
    rich.print(tokenizer.decode(x,skip_special_tokens=True).split("result = ")[-1])

# %% [markdown]
# ## Parse the output
# 
# The output is a Python list of instances, we can execute it  🤯
# 
# We define the AnnotationList class to parse the output with a single line of code. The `AnnotationList.from_output` function filters any label that we did not define (hallucinations) to prevent getting an `undefined class` error. 

# %%
result = AnnotationList.from_output(
    tokenizer.decode(model_ouput[0],skip_special_tokens=True).split("result = ")[-1],
    task_module="guidelines"
    )
rich.print(result)

# %% [markdown]
# Labels are an instance of the defined classes:

# %%
type(result[0])

# %%
result[0].mention

# %% [markdown]
# # Evaluate the result
# 
# Finally, we will evaluate the outputs from the model.

# %% [markdown]
# First, we define an Scorer, for Named Entity Recognition, we will use the `SpanScorer` class.
# 
# We need to define the `valid_types` for the scorer, which will be the labels that we have defined. 

# %%
from src.tasks.utils_scorer import TemplateScorer

class MyScorer(TemplateScorer):
    """Compute the F1 score for Generic Task"""

    valid_types: List[Type] = ENTITY_DEFINITIONS

# %% [markdown]
# ### Instanciate the scorer

# %%
scorer = MyScorer()

# %% [markdown]
# ### Compute F1 

# %%
scorer_results = scorer(reference=[gold],predictions=[result])
rich.print(scorer_results)

# %% [markdown]
# GoLLIE has successfully labeled a sentence using a set of labels that were not part of the pretraining dataset 🎉🎉🎉
# 
# GoLLIE will perform well on labels with well-defined and clearly bounded guidelines. 
# 
# Please share your cool experiments with us; we'd love to see what everyone is doing with GoLLIE!
# - [@iker_garciaf](https://twitter.com/iker_garciaf)
# - [@osainz59](https://twitter.com/osainz59)



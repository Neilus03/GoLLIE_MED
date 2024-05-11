# %% [markdown]
# # GoLLIE for Medical Entity Extraction

# %% [markdown]
# ### Import requeriments
# See the requeriments.txt file in the main directory to install the required dependencies
# 

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

# %% [markdown]
# ## Load GoLLIE
# 
# Load GOLLIE-7B from the huggingface-hub.
# Use the AutoModelForCausalLM.from_pretrained function if you prefer it. However, creators provide a handy load_model function with many functionalities already implemented that will assist you in reproducing our results.
# 
# Please note that setting use_flash_attention=True is mandatory. Our flash attention implementation has small numerical differences compared to the attention implementation in Huggingface. Using use_flash_attention=False will result in the model producing inferior results. Flash attention requires an available CUDA GPU. Running GOLLIE pre-trained models on a CPU is not supported.
# 
# - Set force_auto_device_map=True to automatically load the model on available GPUs.
# - Set quantization=4 if the model doesn't fit in your GPU memory.

# %%
#Use the custom load_model for loading GoLLIE
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
# We will define several classes: `Illness`, `Medication`, `PatientData`, `HospitalizationData`. Each class will have a definition and a set of slots that the model needs to fill. Each slot also requires a type definition and a short description, which can include examples. For instance, for the `Illness` class, we define three slots:
# 
# - The `mention`, which will be the name of the Ilness of the patient and should be a string.
# - The `treatment` which will be a list of treatments or interventions used to manage the illness. 
# - The `symptoms`, which is defined as a list of symptoms. Therefore, GoLLIE will fill this slot with a list of strings.
# 

# %%
from typing import List

from src.tasks.utils_typing import dataclass
from src.tasks.utils_typing import Generic as Template
from dataclasses import dataclass, field


"""
Entity definitions
"""


@dataclass
class Medication(Template):
    """Refers to a drug or substance used to diagnose, cure, treat, or prevent disease.
    Medications can be administered in various forms and dosages and are crucial 
    in managing patient health conditions. They can be classified based on their 
    therapeutic use, mechanism of action, or chemical characteristics."""
    
    mention: str
    """
    The name of the medication.
    Such as: "Aspirina", "Ibuprofeno", "Aspirina".
    """
    dosage: str # The amount and frequency at which the medication is prescribed. Such as: "100 mg al día", "200 mg dos veces al día"
    route: str # The method of administration for the medication. Such as: "oral", "intravenoso", "tópico"
    purpose: List[str]  # List of reasons or conditions for which the medication is prescribed. Such as: ["dolor", "control de azúcar en la sangre", "inflamación"]
    

@dataclass
class Ilness(Template):
    """Refers to a health condition or disease that affects the body's normal functioning.
    Illnesses can be caused by various factors, such as infections, genetic disorders,
    lifestyle choices, or environmental factors. They can affect different body systems
    and have varying degrees of severity."""
    
    mention: str
    """
    The name of the illness or health condition.
    Such as: "diabetes", "cáncer", "hipertensión".
    """
    symptoms: List[str] # List of signs or symptoms associated with the illness. Such as: ["dolor de cabeza", "fatiga", "fiebre"]
    treatment: List[str] # List of treatments or interventions used to manage the illness. Such as: ["medicamentos", "cirugía", "terapia física"]


@dataclass
class HospitalizationData:
    """Refers to information related to a patient's hospitalization, including the
    admission date, discharge date, and reason for hospitalization. Hospitalization
    data is essential for tracking patient health status, treatment progress, and
    healthcare resource utilization."""
    
    admission_date: str #The date on which the patient was admitted to the hospital.
    discharge_date: str #The date on which the patient was discharged from the hospital.
    reason: str #the reason or cause for the patient's hospitalization.
    
    
@dataclass
class PatientData:
    """Refers to information related to a patient's medical history, including
    name, age or urgency. Patient data is essential for healthcare providers 
    to provide appropriate care and make informed decisions about patient management."""
    
    name: str #The name of the patient.
    age: int #The age of the patient.
    urgency: str #The urgency level of the patient's condition.
    
    
    
ENTITY_DEFINITIONS: List[Template] = [
    Medication,
    Ilness,
    HospitalizationData,
]

'''from IPython import In

if __name__ == "__main__":
    print("Entities loaded successfully")
    cell_txt = In[-1] #In needs to be imported from IPython'''

cell_txt = '''from typing import List

from src.tasks.utils_typing import dataclass
from src.tasks.utils_typing import Generic as Template
from dataclasses import dataclass, field


"""
Entity definitions
"""


@dataclass
class Medication(Template):
    """Refers to a drug or substance used to diagnose, cure, treat, or prevent disease.
    Medications can be administered in various forms and dosages and are crucial 
    in managing patient health conditions. They can be classified based on their 
    therapeutic use, mechanism of action, or chemical characteristics."""
    
    mention: str
    """
    The name of the medication.
    Such as: "Aspirina", "Ibuprofeno", "Aspirina".
    """
    dosage: str # The amount and frequency at which the medication is prescribed. Such as: "100 mg al día", "200 mg dos veces al día"
    route: str # The method of administration for the medication. Such as: "oral", "intravenoso", "tópico"
    purpose: List[str]  # List of reasons or conditions for which the medication is prescribed. Such as: ["dolor", "control de azúcar en la sangre", "inflamación"]
    

@dataclass
class Ilness(Template):
    """Refers to a health condition or disease that affects the body's normal functioning.
    Illnesses can be caused by various factors, such as infections, genetic disorders,
    lifestyle choices, or environmental factors. They can affect different body systems
    and have varying degrees of severity."""
    
    mention: str
    """
    The name of the illness or health condition.
    Such as: "diabetes", "cáncer", "hipertensión".
    """
    symptoms: List[str] # List of signs or symptoms associated with the illness. Such as: ["dolor de cabeza", "fatiga", "fiebre"]
    treatment: List[str] # List of treatments or interventions used to manage the illness. Such as: ["medicamentos", "cirugía", "terapia física"]


@dataclass
class HospitalizationData:
    """Refers to information related to a patient's hospitalization, including the
    admission date, discharge date, and reason for hospitalization. Hospitalization
    data is essential for tracking patient health status, treatment progress, and
    healthcare resource utilization."""
    
    admission_date: str #The date on which the patient was admitted to the hospital.
    discharge_date: str #The date on which the patient was discharged from the hospital.
    reason: str #the reason or cause for the patient's hospitalization.
    
    
@dataclass
class PatientData:
    """Refers to information related to a patient's medical history, including
    name, age or urgency. Patient data is essential for healthcare providers 
    to provide appropriate care and make informed decisions about patient management."""
    
    name: str #The name of the patient.
    age: int #The age of the patient.
    urgency: str #The urgency level of the patient's condition.
    
    
    
ENTITY_DEFINITIONS: List[Template] = [
    Medication,
    Ilness,
    HospitalizationData,
]'''


# %% [markdown]
# ### Print the guidelines to guidelines.py
# 
# Due to IPython limitations, we must write the content of the previous cell to a file and then import the content from that file.

# %%
with open("guidelines.py","w",encoding="utf8") as python_guidelines:
    print(cell_txt,file=python_guidelines)

from guidelines import *    ### Print the guidelines to guidelines.py


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
import json
#import gt dataset
data = json.load(open('/hhome/nlp2_g09/Project/uab_summary_2024_with_gt.json'))

# %%
data[0]

# %%
text = data[0]['Text']
print(len(text))
gold = []

# %% [markdown]
# ## Filling a template
# 
# We need to define a template. For this task, we will include only the class definitions and the text to be annotated. However, we can design different templates to incorporate more information (for example, event triggers, as demonstrated in the Event Extraction notebook).
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
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(1))  # Check if device 1 is available
device = "cuda:1" if torch.cuda.is_available() else "cpu"

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

# %% [markdown]
# result = AnnotationList.from_output(
#     tokenizer.decode(model_ouput[0],skip_special_tokens=True).split("result = ")[-1],
#     task_module="guidelines"
#     )
# rich.print(result)

# %%
type(result[0])

# %%
result[0].mention



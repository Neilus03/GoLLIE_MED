import sys
import rich
import logging
from typing import List
import inspect
from jinja2 import Template as jinja2Template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Append the GoLLIE base directory to sys path
sys.path.append("../")

# Load model function from src
from src.model.load_model import load_model
from src.tasks.utils_typing import AnnotationList

from src.tasks.utils_typing import dataclass
from src.tasks.utils_typing import Generic as Template
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)





import torch

# Ensure PyTorch uses the correct CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Load GoLLIE model and tokenizer
model, tokenizer = load_model(
    inference=True,
    model_weights_name_or_path="HiTZ/GoLLIE-7B",
    quantization=None,
    use_lora=False,
    force_auto_device_map=True,
    use_flash_attention=True,
    torch_dtype="bfloat16"
)

# Ensure the model is on the correct device
model.to(device)

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

# Use inspect.getsource to get the guidelines as a string
guidelines = [inspect.getsource(definition) for definition in ENTITY_DEFINITIONS]

# Define input text (replace this with your actual text)
text = """
El paciente fue ingresado al hospital el 15 de enero de 2022 con un cuadro de dolor abdominal intenso que acabó en cancer de colon, sus sintomas son dolor de ovario, el tratamiento será con quimioterapia.
"""

# Define the template for the prompt
template_txt = (
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

# Create the Jinja2 template
template = jinja2Template(template_txt)

# No gold annotations available, so we provide an empty list
gold = []

# Fill the template
formatted_text = template.render(guidelines=guidelines, text=text, annotations=gold, gold=gold)

# Use Black to format the code
import black
black_mode = black.Mode()
formatted_text = black.format_str(formatted_text, mode=black_mode)

# Print the filled and formatted template
rich.print(formatted_text)

# Prepare model inputs
prompt, _ = formatted_text.split("result =")
prompt = prompt + "result ="

# Tokenize the input sentence
model_input = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

# Remove the `eos` token from the input
model_input["input_ids"] = model_input["input_ids"][:, :-1]
model_input["attention_mask"] = model_input["attention_mask"][:, :-1]

# Run GoLLIE
model_output = model.generate(
    **model_input.to(model.device),
    max_new_tokens=128,
    do_sample=False,
    min_new_tokens=0,
    num_beams=1,
    num_return_sequences=1,
)

# Print the results
for y, x in enumerate(model_output):
    print(f"Answer {y}")
    rich.print(tokenizer.decode(x, skip_special_tokens=True).split("result = ")[-1])

# Parse the output
result = AnnotationList.from_output(
    tokenizer.decode(model_output[0], skip_special_tokens=True).split("result = ")[-1],
    task_module="guidelines"
)
rich.print(result)

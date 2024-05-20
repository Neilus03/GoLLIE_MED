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
    
    name: str #The name and surname of the patient.
    age: int #The age of the patient.
    urgency: str #The urgency level of the patient's condition.
    
    
    
ENTITY_DEFINITIONS: List[Template] = [
    Medication,
    Ilness,
    HospitalizationData,
    PatientData
]

from typing import List
from src.tasks.utils_typing import dataclass
from src.tasks.utils_typing import Generic as Template
from dataclasses import dataclass, field


"""
Definiciones de entidades
"""

@dataclass
class Medicación:
    """Se refiere a un fármaco o sustancia utilizada para diagnosticar, curar, tratar o prevenir enfermedades.
    Las medicaciones pueden administrarse en diversas formas y dosis y son cruciales 
    para el manejo de las condiciones de salud del paciente. Pueden clasificarse 
    según su uso terapéutico, mecanismo de acción o características químicas."""
    
    mención: str
    """
    El nombre de la medicación.
    Por ejemplo: "Aspirina", "Ibuprofeno", "Aspirina".
    """
    dosis: str # La cantidad y frecuencia con la que se prescribe la medicación. Por ejemplo: "100 mg al día", "200 mg dos veces al día".
    vía: str # El método de administración de la medicación. Por ejemplo: "oral", "intravenoso", "tópico".
    propósito: List[str]  # Lista de razones o condiciones para las que se prescribe la medicación. Por ejemplo: ["dolor", "control de azúcar en la sangre", "inflamación"].
    fecha_inicio: str # La fecha en la que se comenzó a tomar la medicación. Por ejemplo: "01-01-2023".
    fecha_fin: str # La fecha en la que se dejó de tomar la medicación, si aplica. Por ejemplo: "31-12-2023".

@dataclass
class Enfermedad:
    """Se refiere a una condición de salud o enfermedad que afecta el funcionamiento normal del cuerpo.
    Las enfermedades pueden ser causadas por diversos factores, como infecciones, trastornos genéticos,
    elecciones de estilo de vida o factores ambientales. Pueden afectar diferentes sistemas del cuerpo
    y tener distintos grados de severidad."""
    
    mención: str
    """
    El nombre de la enfermedad o condición de salud.
    Por ejemplo: "diabetes", "cáncer", "hipertensión".
    """
    síntomas: List[str] # Lista de signos o síntomas asociados con la enfermedad. Por ejemplo: ["dolor de cabeza", "fatiga", "fiebre"].
    tratamiento: List[str] # Lista de tratamientos o intervenciones utilizadas para manejar la enfermedad. Por ejemplo: ["medicamentos", "cirugía", "terapia física"].
    fecha_diagnostico: str # La fecha en la que se diagnosticó la enfermedad. Por ejemplo: "15-05-2022".
    severidad: str # El nivel de severidad de la enfermedad. Por ejemplo: "leve", "moderado", "grave".

@dataclass
class ProcedimientoMedico:
    """Se refiere a las intervenciones médicas realizadas para diagnosticar o tratar enfermedades.
    Esto puede incluir cirugías, pruebas diagnósticas y otros tratamientos especializados."""
    
    mención: str
    """
    El nombre del procedimiento médico.
    Por ejemplo: "biopsia", "resonancia magnética", "cirugía laparoscópica".
    """
    fecha: str # La fecha en la que se realizó el procedimiento. Por ejemplo: "10-02-2023".
    resultado: str # El resultado o conclusión del procedimiento. Por ejemplo: "negativo para malignidad".

@dataclass
class DatosHospitalización:
    """Se refiere a la información relacionada con la hospitalización de un paciente, incluida la
    fecha de ingreso, fecha de alta y motivo de la hospitalización. Los datos de hospitalización
    son esenciales para rastrear el estado de salud del paciente, el progreso del tratamiento y
    la utilización de recursos sanitarios."""
    
    fecha_ingreso: str # La fecha en la que el paciente fue ingresado en el hospital.
    fecha_alta: str # La fecha en la que el paciente fue dado de alta del hospital.
    motivo: str # El motivo o causa de la hospitalización del paciente.
    unidad: str # La unidad o departamento del hospital en el que el paciente estuvo internado. Por ejemplo: "Unidad de Cuidados Intensivos".
    medico_responsable: str # El nombre del médico responsable del paciente durante la hospitalización.

@dataclass
class DatosPaciente:
    """Se refiere a la información relacionada con el historial médico de un paciente, incluido
    el nombre, la edad o la urgencia. Los datos del paciente son esenciales para que los proveedores
    de atención médica brinden el cuidado adecuado y tomen decisiones informadas sobre el manejo del paciente."""
    
    nombre: str # El nombre del paciente.
    edad: int # La edad del paciente.
    urgencia: str # El nivel de urgencia de la condición del paciente.
    sexo: str # El sexo del paciente. Por ejemplo: "hombre", "mujer".
    fecha_nacimiento: str # La fecha de nacimiento del paciente. Por ejemplo: "12-07-1962".
    antecedentes_personales: List[str] # Lista de antecedentes médicos personales relevantes. Por ejemplo: ["hipertensión", "diabetes"].

@dataclass
class SignosVitales:
    """Se refiere a las mediciones de las funciones básicas del cuerpo que son esenciales para la vida.
    Los signos vitales incluyen la temperatura corporal, frecuencia cardíaca, presión arterial, 
    frecuencia respiratoria y saturación de oxígeno."""
    
    temperatura: float # La temperatura corporal del paciente. Por ejemplo: "36.5".
    frecuencia_cardiaca: int # La cantidad de latidos del corazón por minuto. Por ejemplo: "72".
    presion_arterial_sistolica: int # La presión en las arterias cuando el corazón late. Por ejemplo: "120".
    presion_arterial_diastolica: int # La presión en las arterias entre los latidos del corazón. Por ejemplo: "80".
    frecuencia_respiratoria: int # La cantidad de respiraciones por minuto. Por ejemplo: "16".
    saturacion_oxigeno: float # El porcentaje de oxígeno en la sangre. Por ejemplo: "98.0".

@dataclass
class ResultadosLaboratorio:
    """Se refiere a los resultados de las pruebas de laboratorio realizadas durante la hospitalización 
    del paciente. Estas pruebas pueden incluir análisis de sangre, análisis de orina y otros estudios clínicos."""
    
    tipo_prueba: str # El tipo de prueba de laboratorio realizada. Por ejemplo: "análisis de sangre".
    resultados: List[str] # Los resultados específicos de la prueba. Por ejemplo: ["glucosa: 90 mg/dL", "creatinina: 1.2 mg/dL"].
    fecha: str # La fecha en la que se realizaron las pruebas. Por ejemplo: "01-06-2023".

@dataclass
class ImagenDiagnostica:
    """Se refiere a los estudios de imagen realizados para diagnosticar o monitorear condiciones de salud.
    Estos estudios pueden incluir radiografías, tomografías, resonancias magnéticas, entre otros."""
    
    tipo_imagen: str # El tipo de estudio de imagen. Por ejemplo: "radiografía de tórax".
    hallazgos: str # Los hallazgos o conclusiones del estudio de imagen. Por ejemplo: "elevación del hemidiafragma izquierdo".
    fecha: str # La fecha en la que se realizó el estudio de imagen. Por ejemplo: "05-06-2023".

DEFINICIONES_ENTIDAD: List[Template] = [
    Medicación,
    Enfermedad,
    ProcedimientoMedico,
    DatosHospitalización,
    DatosPaciente,
    SignosVitales,
    ResultadosLaboratorio,
    ImagenDiagnostica,
]

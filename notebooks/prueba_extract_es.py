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


# Use inspect.getsource to get the guidelines as a string
guidelines = [inspect.getsource(definition) for definition in DEFINICIONES_ENTIDAD]

# Define input text (replace this with your actual text)
text = """
diagnostic alta codi icd-10 descripcio diagnostic k70.3 cirrosi hepatica alcoholica r18.8 altres tipus d'ascites dades informe motivo de ingreso: mal estado general antecedentes personales: no ram. hta. disnea a pequeños esfuerzos y epoc gold ii en seguimiento en neumologia en tratamiento broncodilatador y oxigenoterapia en concentrador portatil a 3lx' para la deambulacion desde enero 2023. elevacion de hemidiafragma izquierdo y adenopatia hiliar derecha con captacion en pet-tc estudiado con ebus + fbs sin lesiones endobronquiales y con ap negativa para malignidad del bas y adenopatias puncionadas. hemorragia digestiva baja secundaria a colitis isquemica en setiembre 2022 (cordoba). colelitiasis. - cirrosis hepatica enolica child c en seguimiento en consulta ch intensivo (dra. martin) **descompensacion edemo-ascitica en junio 2022 con empeoramiento tras ingreso hospitalario por colitis isquemica (setiembre 2022), en tratamiento diuretico con espironolactona 200mg/dia + furosemida 80mg/dia desde el 18.05.23 **encefalopatia hepatica en agosto 2022. en tratamiento con lactulosa + rifaximina desde noviembre 2022 **varices esofagicas pequeñas con signos de riesgo (fgs en setiembre - cordoba) en tratamiento con carvedilol 6.25mg/12horas **ultima ecografia en marzo 2023: no loes hepaticas, porta permeable,discreta cantidad de liquido libre perihepatico y en pelvis ex enolismo cronico, abstinente desde setiembre 2022. fumador 5-6 cig/dia tratamiento habitual: acfol 5mg/dia, carvedilol 6.25mg/12horas, espironolactona 200mg/dia, furosemida 80mg/dia, lactulosa, spiolto 2ing/12horas, iabvd. vive con su madre. enfermedad actual traido por familiares a urgencias por deterioro del estado general, somnolencia, astenia y aumento de edemas en miembros inferiores a pesar de aumento de diureticos. refiere tambien que hace una semana cursa con deposiciones blandas (3-4 deposiciones/dia) sin productos patologicos motivo por el cual suspendio lactulosa. no fiebre, no sintomas urinarios, no dolor abdominal. hoy empeoramiento de su disnea habitual. exploracion fisica urgencias: tº 35.9, tas 150, tad 65, sato2 bas 94, fr 16, fc 57 estado general conservado, piel y mucosas hidratadas levemente ictericas. nrl: consciente, orientado en tiempo, espacio y persona, discurso fluido y coherente, no flapping. ac: ruidos cardiacos ritmicos sin soplos. ar: crepitantes basales bilaterales e hipofonesis basal izquierda. ************* pacient cip data naix. 12.07.1962 edat 60 sexe home nass adreça cp poblacio tel. ****************** admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22 ***************************************** *************************************************************** ***************** pagina 1 de 4 informe alta hospitalitzacio abdomen globuloso, blando, depresible, no palpo masas ni megalias, rha conservados, no dolor a la palpacion, sin reaccion peritoneal, edema de pared. eeii: presenta edemas en eeii hasta muslos, fovea ++. planta digestivo ta 143/74 fc72 afebril. sato2 basal 90% ap: hipofonesis en base izquierda. abd: globuloso, no clara semiologia ascitica. no irritacion peritoneal. rha + ext: edemas ++/+++ hasta muslos sn: consciente y orientado. no flapping pruebas complementarias: + analitica: leucocitos 5740 (n 3220, l 1600), hb 123 g/l, plaquetas 140000, tp 49%, inr 1.64, ttpa 1.55, glucosa 48mg/dl, creatinina 1.7 mg/dl, fg 43.98 ml/min, urea 22.1 mg/dl, sodio 139.0 mmol/l, potasio 4.0 mmol/l, alt 19.8 ui/l, albumina 1.5 g/dl, pcr 2.79 mg/l, + eab (arterial): ph 7.366, co2 24.6 mmhg, o2 65.8 mmhg, bic 13.8 mmol/l, be -9.1, sat. 89.8%. + sedimento urinario: eritrocitos 20-50/c, leucocitos 3-5/c. + rx torax : elevacion de hemidiafragma izquierdo, borramiento de angulo costofrenico izquierdo. - paracentesis x3 : blanca - analitica 01.06: leucocitos 3910, hb 10.9 g/dl, hto 30%, vcm 92, plaquetas 108000, tp 43%, glucosa 133 mg/dl, na 143, k 4.2, br 3.77 mg/dl, ast/alt 37/21, fa/ggt 82/15, pcr 1. - hemocultivo: negativo - coprocultivo: negativo - adenovirus y rotavirus en heces: negativo - gdh c. difficile en heces: negativo - pcr covid: negativo - ecocardiograma : fevi conservado. hallazgos compatibles con cardiopatia hipertensiva. - urocultivo: negativo - gammagrafia : no hay imagenes sugestivas de corresponder con tep en la gpp obtenida y, en todo caso, valoramos el estudio indeterminado en localizacion de la base pulmonar izquierda donde hay una hipoperfusion moderada no segmentaria que coincide con alteracion radiologica (atelectasia/colapso basal sin descartar sobreinfeccion asociada?? y elevacion de este hemidiafragma). - cateterismo hepatico y cardiopulmonar: cateterismo de venas suprahepaticas por via yugular. obtencion de presiones hepaticas mediante cateter balon. mediciones repetidas de la presion suprahepatica enclavada (pse) y del gradiente de presion venosa hepatica (gpvh). resultados (mmhg) pse:31.0 psl:15.5 gpvh: 15.5 pvci: 15.5 pad: 13.0 2. cateterismo cardiopulmonar. pap: 32.0 mm hg pcp: 28.0 mm hg pad: 13.0 mm hg gc: 10.1 l/min pam: 115 mm hg fc: 74 latidos/min ic: 4.9 l/min/m2 sv: 136 ml/latido rvs: 808 dynas*seg*cm-5 irvs: 1665 dynas*seg*cm-5 rvp: 32 dynas*seg*cm-5 irvp: 65 dynas*seg*cm-5 pap al final de la espiracion: 35.0 mm hg ************* pacient cip data naix. 12.07.1962 edat 60 sexe home nass adreça cp poblacio tel. ****************** admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22 ***************************************** *************************************************************** ***************** pagina 2 de 4 informe alta hospitalitzacio conclusiones 1.- cateterismo cardiopulmonar con presiones cardiopulmonares elevadas (pap 32 mmhg, pcp 28 mmhg, ad 13mmhg). gasto cardiaco de 10,1 l/min). 2.- hipertension portal sinusoidal clinicamente significativa con gpvh de 15,5mmhg, no se observan comunicantes veno-venosas. - orina de 24horas: total diuresis: 1.4 l/dia. albumina 5125 mg/dia proteinuria 7.39 g/24horas orientacion diagnostica - insuficiencia renal aguda en paciente con enfermedad renal cronica en estudio paciente que en analitica de ingreso tenia cr de 1.7 mg/dl, fg 43 ml/min cuando previamente tenia filtrados habituales de 70 ml/min, cr 1.1 mg/dl. se oriento como de origen prerrenal en contexto de aumento de la dosis de diureticos (estaba con espironolactona 200mg/dia + furosemida 80mg/dia ). recibio expansion con albumina 1 g/kg (80gr) el dia 31.05 y posteriormente 2 ampollas/ 12 h ev sin mayor mejoria de la funcion renal a pesar de reexpansion con albumina por lo que se suspendio albumina que se vuelve a reiniciar el dia 8/06 y se mantiene durante otras 72h. en contexto de sobrecarga de volumen el paciente cursa con empeoramiento de disnea habitual por lo que se retira albumina y se inicia tratamiento con furosemida iv 40mg/12horas dado que podria tratarse de una erc establecida se solicita estudio de orina de 24horas en la que se observa proteinuria en rangos nefroticos por lo que ha sido valorado por nefrologia, se inicia tratamiento con espironolactona 100mg/dia y furosemida 80mg/dia via oral. al alta con creat 1.8 y fg 39 - descompensacion edemo-ascitica . cirrosis hepatica enolica child c10 meld 21 puntos al ingreso con edemas que empeoran durante la hospitalizacion debido a que se suspendieron diureticos por insuficiencia renal. se reinicia diureticos debido a sobrecarga de volumen con mejoria de los edemas en miembros inferiores y disminucion del peso. se ha intentado en 3 ocasiones realizar paracentesis diagnosticas que han sido blancas. al alta con espironolactona 100mg/dia + furosemida 80mg/dia y peso de 87kg. - insuficiencia respiratoria cronica mixta : sobrecarga de volumen + restrictivo. epoc gold ii. al ingreso disnea al reposo y desaturacion por lo que ha recibido tratamiento con broncodilatadores inhalados y o2 por vmk 26%. valorado por neumologia, el grado de epoc que presenta no justifica la disnea a minimos esfuerzos y la desaturacion. en este ingreso, se solicita gammagrafia v/ p que descarta tep cronico. se ha descartado sd hepato-pulmonar por ecocardiograma con contraste burbujas que no visualiza shunts. cateterismo cardio pulmonar que descarta sde portopulmonar . elevacion hemidiafragma izquierdo que provoca atelectasia pulmonar 2º. pdte de informe de tac toracico (06.06). tras inicio de furosemida iv el paciente presenta franca mejoria de disnea y de saturacion. al alta sat o2 93-94% al aire ambiental. seguira controles en consulta de neumologia. - fiebre sin foco. resuelto el 8/06 presento pico febril asociado a desaturacion puntual. se solicitan hemocultivos, urocultivo y se realiza paracentesis que resulta blanca. se inicia cobertura atb con ceftriaxona 1gr/dia durante 5 dias. recomendaciones al alta - dieta baja en sal ************* pacient cip data naix. 12.07.1962 edat 60 sexe home nass adreça cp poblacio tel. ****************** admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22 ***************************************** *************************************************************** ***************** pagina 3 de 4 informe alta hospitalitzacio - abstinencia absoluta del alcohol - control de peso diario - furosemida 40mg: 1-1-0 - espironolactona 100mg: 1-0-0 - lactulosa: 1 sobre al dia. suspender si diarrea - continuar con oxigeno a 3lx' para la deambulacion - resto de medicacion habitual - control y seguimiento en consulta ch intensivo (dra. martin) y en especialistas habituales tipus d'ingres: urgent motiu d'alta: alt.med.domicil metge adjunt: *****************, ***************** ****************************** servei: diguohmb digestologia data informe: 16.06.2023 ************* pacient cip data naix. 12.07.1962 edat 60 sexe home nass adreça cp poblacio tel. ****************** admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22 ***************************************** *************************************************************** ***************** pagina 4 de 4 informe alta hospitalitzacio"""

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
    max_new_tokens=2048,
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

#save the output
result.to_json("output.json")

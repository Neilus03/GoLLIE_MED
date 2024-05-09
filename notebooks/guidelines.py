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
    
if __name__ == "__main__":
    cell_txt = In[-1]

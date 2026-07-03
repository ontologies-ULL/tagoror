from abc import ABC, abstractmethod
from owlready2 import Thing, Ontology

class BaseSerializer(ABC):
    """
    Abstract base class for serializers that convert OWL individuals and ontologies into specific formats.
    """
    
    @abstractmethod
    def process_ontology(self, ontology: Ontology) -> str:
        pass

    @abstractmethod
    def process_individual(self, individual: Thing) -> str:
        pass
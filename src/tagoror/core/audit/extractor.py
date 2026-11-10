"""
Extract ontology individuals into structured records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from owlready2 import get_ontology, Thing 

class OntologyExtractor:
    """
    Converts ontology individuals into OntologyExtractionRecord instances.
    """

    _base_ontology_path: Path = Path(__file__).parent.parent.parent / "base_ontologies" / "base_ontology.rdf"
    _base_ontology = None
    _base_names = set()

    @classmethod
    def extract(cls, file_path: str) -> list[Thing]:
        """
        Extract only ontology individuals.
        """
        cls._initialize_base_ontology()
        ontology = OntologyExtractor._load_ontology(Path(file_path).resolve().as_uri())        
        result = []
        for individual in ontology.individuals():
            if str(individual.name) not in cls._base_names:
                result.append(individual)
        return result   

    @classmethod
    def get_base_ontology(cls) -> Any:
        cls._initialize_base_ontology()    
        return cls._base_ontology
    
    @classmethod
    def _initialize_base_ontology(cls):
        if cls._base_ontology is None:
            cls._base_ontology = cls._load_ontology(cls._base_ontology_path.resolve().as_uri())
        if cls._base_names is None or not cls._base_names:
            cls._base_names = {str(ind.name) for ind in cls._base_ontology.individuals() if ind.name is not None}

    @staticmethod
    def _load_ontology(file_path: str, ontology_loader=None):
        """
        Load an ontology file and return the parsed ontology object.
        """
        if ontology_loader is None:
            ontology_loader = get_ontology
        return ontology_loader(file_path).load()
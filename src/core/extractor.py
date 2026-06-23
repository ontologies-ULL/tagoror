"""
Extract ontology individuals into structured records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from owlready2 import get_ontology 

class OntologyExtractor:
  """
  Converts ontology individuals into OntologyExtractionRecord instances.
  """

  _base_ontology_path: Path = Path(__file__).parent / "base_ontology.ttl"
  _base_ontology = None

  @classmethod
  def extract(cls, file_path: str) -> list[Any]:
    """
    Extract only ontology individuals.
    """
    if cls._base_ontology is None:
      cls._base_ontology = cls._load_ontology(cls._base_ontology_path)

    ontology = OntologyExtractor._load_ontology(file_path)
    return [individual for individual in ontology.individuals() if individual not in OntologyExtractor._base_ontology.individuals()]

  @staticmethod
  def _load_ontology(file_path: str, ontology_loader=None):
    """
    Load an ontology file and return the parsed ontology object.
    """
    if ontology_loader is None:
      ontology_loader = get_ontology
    return ontology_loader(file_path).load()
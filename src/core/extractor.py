"""
Extract ontology individuals into structured records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from owlready2 import get_ontology, ObjectProperty


@dataclass(frozen=True)
class OntologyExtractionRecord:
  """
  Normalized record for one ontology individual.
  """

  CaseID: str
  Individual: str
  Classes: list[str] = field(default_factory=list)
  ObjectProperties: dict[str, list[Any]] = field(default_factory=dict)
  DataProperties: dict[str, list[Any]] = field(default_factory=dict)
  Annotations: dict[str, list[Any]] = field(default_factory=dict)


class OntologyExtractor:
  """
  Converts ontology individuals into OntologyExtractionRecord instances.
  """

  @staticmethod
  def extract(file_path: str) -> list[OntologyExtractionRecord]:
    """
    Extract only ontology individuals.
    """
    ontology = OntologyExtractor._load_ontology(file_path)
    case_id = OntologyExtractor._build_case_id(file_path)

    return [
      OntologyExtractor._build_record(individual, case_id)
      for individual in ontology.individuals()
    ]

  @staticmethod
  def _load_ontology(file_path: str, ontology_loader=None):
    """
    Load an ontology file and return the parsed ontology object.
    """
    if ontology_loader is None:
      ontology_loader = get_ontology
    return ontology_loader(file_path).load()

  @staticmethod
  def _build_case_id(file_path: str) -> str:
    """
    Build the CaseID value from the file name.
    """
    return Path(file_path).name

  @staticmethod
  def _extract_classes(individual) -> list[str]:
    """
    Collect the class names attached to an ontology individual.
    """
    return [
      cls.name
      for cls in getattr(individual, "is_a", [])
      if getattr(cls, "name", None)
    ]

  @staticmethod
  def _normalize_values(values: Any, resolve_names: bool = False) -> list[Any]:
    """
    Normalize property values into a list.
    """
    if isinstance(values, (list, tuple, set)):
      normalized_values = list(values)
    else:
      normalized_values = [values]
    if resolve_names:
      return [OntologyExtractor._resolve_name(value) for value in normalized_values]

    return normalized_values

  @staticmethod
  def _resolve_name(value: Any) -> Any:
    """
    Convert ontology references into readable names when possible.
    """
    if hasattr(value, "name") and getattr(value, "name", None):
      return value.name

    return value

  @staticmethod
  def _is_object_property(property_descriptor) -> bool:
    """
    Detect whether a property descriptor is an object property.
    Works with both real OWL properties and mock objects in tests.
    """
    if isinstance(property_descriptor, type):
        return issubclass(property_descriptor, ObjectProperty)
    
    return bool(
        getattr(property_descriptor, 'is_object_property', False) or 
        getattr(property_descriptor, 'kind', '') == 'object_property'
    )

  @staticmethod
  def _extract_object_properties(individual) -> dict[str, list[Any]]:
    """
    Collect object properties for one ontology individual.
    """
    object_properties: dict[str, list[Any]] = {}

    for property_descriptor in individual.get_properties():
      if not OntologyExtractor._is_object_property(property_descriptor):
        continue

      property_name = property_descriptor.name
      property_values = getattr(individual, property_name)
      object_properties[property_name] = OntologyExtractor._normalize_values(
        property_values,
        resolve_names=True,
      )

    return object_properties

  @staticmethod
  def _extract_data_properties(individual) -> dict[str, list[Any]]:
    """
    Collect the data properties for one ontology individual.
    """
    data_properties: dict[str, list[Any]] = {}

    for property_descriptor in individual.get_properties():
      if OntologyExtractor._is_object_property(property_descriptor):
        continue

      property_name = property_descriptor.name
      property_values = getattr(individual, property_name)
      data_properties[property_name] = OntologyExtractor._normalize_values(property_values)

    return data_properties

  @staticmethod
  def _extract_annotations(individual) -> dict[str, list[Any]]:
    """
    Collect annotations for one ontology individual when they are available.
    """
    annotations: dict[str, list[Any]] = {}
    
    for attr_name in ["label", "comment", "isDefinedBy", "seeAlso"]:
        values = getattr(individual, attr_name, [])
        if values:
            annotations[attr_name] = OntologyExtractor._normalize_values(values)
            
    return annotations

  @staticmethod
  def _build_record(individual, case_id: str) -> OntologyExtractionRecord:
    """
    Build a record from a single ontology individual.
    """
    return OntologyExtractionRecord(
      CaseID=case_id,
      Individual=individual.name,
      Classes=OntologyExtractor._extract_classes(individual),
      ObjectProperties=OntologyExtractor._extract_object_properties(individual),
      DataProperties=OntologyExtractor._extract_data_properties(individual),
      Annotations=OntologyExtractor._extract_annotations(individual),
    )

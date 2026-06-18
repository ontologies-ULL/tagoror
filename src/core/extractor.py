"""
Extract ontology individuals into validation entities.
"""

from typing import Any

from validation.models import ExtractedEntity
from owlready2 import get_ontology


def extract_entities(file_path: str, ontology_loader=None) -> list[ExtractedEntity]:
  """
  Load an ontology file and extract its individuals.
  """
  if ontology_loader is None:
    ontology_loader = get_ontology

  try:
    ontology = ontology_loader(file_path).load()
  except Exception as error:
    raise FileNotFoundError(file_path) from error

  extracted_entities: list[ExtractedEntity] = []

  for individual in ontology.individuals():
    classes = [cls.name for cls in getattr(individual, "is_a", []) if getattr(cls, "name", None)]
    data_properties: dict[str, Any] = {}

    for property_descriptor in individual.get_properties():
      property_name = property_descriptor.name
      property_values = getattr(individual, property_name)

      if isinstance(property_values, list):
        normalized_values = property_values
      elif isinstance(property_values, tuple):
        normalized_values = list(property_values)
      elif isinstance(property_values, set):
        normalized_values = list(property_values)
      else:
        normalized_values = [property_values]

      data_properties[property_name] = normalized_values

    extracted_entities.append(
      ExtractedEntity(
        individual_id=individual.name,
        classes=classes,
        data_properties=data_properties,
      )
    )

  return extracted_entities


class OntologyExtractor:
  """
  Converts ontology individuals into ExtractedEntity instances.
  """

  @staticmethod
  def extract(file_path: str) -> list[ExtractedEntity]:
    """
    Load an ontology file and extract its individuals.
    """
    return extract_entities(file_path)

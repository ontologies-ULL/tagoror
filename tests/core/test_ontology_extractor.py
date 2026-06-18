"""
Unit tests for OntologyExtractor
================================

Contract covered here:
  - extract(file_path) returns only ontology individuals
  - records use the requested names: CaseID, Individual, Classes,
    ObjectProperties, DataProperties, Annotations
  - missing/invalid file raises FileNotFoundError
"""

import pytest
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_class(name: str):
    return SimpleNamespace(name=name)

def make_fake_property(name: str, kind: str):
    return SimpleNamespace(
        name=name,
        is_object_property=(kind == "object"),
        kind="object_property" if kind == "object" else "data_property",
        property_kind="object" if kind == "object" else "data"
    )

def make_fake_annotation(name: str):
    return SimpleNamespace(name=name)

def make_fake_entity(name: str, classes=None):
    return SimpleNamespace(
        name=name,
        is_a=[make_fake_class(class_name) for class_name in (classes or [])]
    )

def make_fake_individual(name: str, classes=None, object_properties=None, data_properties=None, annotations=None):
    classes = classes or []
    object_properties = object_properties or {}
    data_properties = data_properties or {}
    annotations = annotations or {}

    individual = make_fake_entity(name, classes=classes)
    property_descriptors = []

    for property_name, values in object_properties.items():
        property_descriptors.append(make_fake_property(property_name, "object"))
        setattr(individual, property_name, values)

    for property_name, values in data_properties.items():
        property_descriptors.append(make_fake_property(property_name, "data"))
        setattr(individual, property_name, values)

    individual.get_properties = lambda: property_descriptors
    annotation_descriptors = []
    
    for annotation_name, values in annotations.items():
        annotation_descriptors.append(make_fake_annotation(annotation_name))
        setattr(individual, annotation_name, values)

    individual.get_annotations = lambda: annotation_descriptors

    return individual

def make_fake_ontology(individuals=None):
    return SimpleNamespace(
        individuals=lambda: individuals or []
    )

# ---------------------------------------------------------------------------
# Tests: extract() -> individuals only
# ---------------------------------------------------------------------------

class TestExtractIndividuals:

    def test_returns_empty_list_for_empty_ontology(self, mocker):
        onto = make_fake_ontology()
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("case.owl")

        assert records == []

    def test_returns_only_individual_records(self, mocker):
        onto = make_fake_ontology(
            individuals=[
                make_fake_individual(
                    "ind_1",
                    classes=["ManifestacionClinica"],
                    object_properties={"relatedTo": [make_fake_entity("target_1")]},
                    data_properties={"hasProbability": [0.78], "hasTag": ("acute", "severe")},
                    annotations={"label": ["Fiebre"]},
                )
            ]
        )
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("/tmp/case.owl")

        assert len(records) == 1
        record = records[0]
        assert record.CaseID == "case.owl"
        assert record.Individual == "ind_1"
        assert record.Classes == ["ManifestacionClinica"]
        assert record.ObjectProperties == {"relatedTo": ["target_1"]}
        assert record.DataProperties == {"hasProbability": [0.78], "hasTag": ["acute", "severe"]}
        assert record.Annotations == {"label": ["Fiebre"]}

    def test_preserves_individual_order(self, mocker):
        onto = make_fake_ontology(
            individuals=[
                make_fake_individual("ind_A"),
                make_fake_individual("ind_B"),
            ]
        )
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("case.owl")

        assert [record.Individual for record in records] == ["ind_A", "ind_B"]

    def test_missing_file_raises_file_not_found(self, mocker):
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.side_effect = FileNotFoundError("missing")

        from core.extractor import OntologyExtractor

        with pytest.raises(FileNotFoundError):
            OntologyExtractor.extract("missing.owl")

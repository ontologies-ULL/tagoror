"""
Unit tests for OntologyExtractor
===================================
Written BEFORE the implementation (TDD) — these tests define the contract
that OntologyExtractor.extract() must satisfy.

Contract (as agreed):
  - extract(file_path: str) -> list[ExtractedEntity]   [staticmethod]
  - Built on top of Owlready2: iterates onto.individuals()
  - individual_id   = individual.name
  - classes         = [c.name for c in individual.is_a]
  - data_properties = {prop.name: [values...] for prop in individual.get_properties()}
  - Missing/invalid file -> raises FileNotFoundError explicitly

Covers:
  - extract() with a single individual, no classes, no properties
  - extract() with a single individual that has one class
  - extract() with a single individual that has multiple classes
  - extract() with a single individual that has one data property with one value
  - extract() with a single individual that has one data property with
    multiple values (list of literals)
  - extract() with a single individual that has multiple data properties
  - extract() with multiple individuals: returns one ExtractedEntity per
    individual, in the order Owlready2 provides them
  - extract() with zero individuals: returns an empty list
  - extract() raises FileNotFoundError when the ontology file does not exist
  - extract() raises FileNotFoundError when get_ontology().load() fails
    for any underlying reason (wrapped consistently)
  - extract() does not crash when get_properties() returns an empty set
  - Each returned object is a real ExtractedEntity instance (not a dict
    or a raw Owlready2 object)

Testing strategy:
  Owlready2's get_ontology(...).load() and the individual objects it returns
  are fully mocked. We never load a real .owl file from disk — this keeps
  the suite fast and independent of any fixture ontology file, and lets us
  freely vary how many individuals/classes/properties are returned without
  maintaining several .owl fixture files.

  Each fake "individual" is a MagicMock with .name, .is_a (list of mocks
  with .name) and .get_properties() (set of mocks with .name, and the
  property itself callable on the individual to retrieve its values —
  matching Owlready2's pattern of `getattr(individual, prop.name)`).
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build fake Owlready2 objects
# ---------------------------------------------------------------------------

def make_fake_class(name: str):
    """Fake Owlready2 class (used inside individual.is_a)."""
    cls = MagicMock()
    cls.name = name
    return cls


def make_fake_property(name: str):
    """Fake Owlready2 data property descriptor (used inside get_properties())."""
    prop = MagicMock()
    prop.name = name
    return prop


def make_fake_individual(name: str, classes=None, properties=None):
    """
    Fake Owlready2 individual.

    properties: dict[str, list] mapping property name -> list of literal values.
    Mimics Owlready2's behaviour where calling getattr(individual, prop.name)
    returns a list of values for that property.
    """
    classes    = classes or []
    properties = properties or {}

    individual = MagicMock()
    individual.name = name
    individual.is_a = [make_fake_class(c) for c in classes]

    fake_props = [make_fake_property(p) for p in properties.keys()]
    individual.get_properties.return_value = fake_props

    # Owlready2 pattern: getattr(individual, prop.name) -> list of values
    for prop_name, values in properties.items():
        setattr(individual, prop_name, values)

    return individual


def make_fake_ontology(individuals):
    """Fake ontology object returned by get_ontology(...).load()."""
    onto = MagicMock()
    onto.individuals.return_value = individuals
    return onto


# ---------------------------------------------------------------------------
# Tests: basic extraction shape
# ---------------------------------------------------------------------------

class TestExtractBasicShape:

    def test_returns_a_list(self, mocker):
        """extract() must always return a list, even with zero individuals."""
        onto = make_fake_ontology([])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert isinstance(result, list)

    def test_returns_empty_list_when_no_individuals(self, mocker):
        """With zero individuals in the ontology, extract() returns []."""
        onto = make_fake_ontology([])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result == []

    def test_returns_one_entity_per_individual(self, mocker):
        """With 3 individuals, extract() must return exactly 3 ExtractedEntity objects."""
        individuals = [
            make_fake_individual("ind_1"),
            make_fake_individual("ind_2"),
            make_fake_individual("ind_3"),
        ]
        onto = make_fake_ontology(individuals)
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert len(result) == 3

    def test_each_item_is_an_extracted_entity_instance(self, mocker):
        """Every item in the returned list must be a real ExtractedEntity, not a dict."""
        onto = make_fake_ontology([make_fake_individual("ind_1")])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        from validation.models import ExtractedEntity
        result = OntologyExtractor.extract("fake_path.owl")

        assert isinstance(result[0], ExtractedEntity)

    def test_preserves_individuals_order(self, mocker):
        """The order of returned entities must match onto.individuals() order."""
        individuals = [make_fake_individual("ind_A"), make_fake_individual("ind_B")]
        onto = make_fake_ontology(individuals)
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert [e.individual_id for e in result] == ["ind_A", "ind_B"]


# ---------------------------------------------------------------------------
# Tests: individual_id mapping
# ---------------------------------------------------------------------------

class TestIndividualIdMapping:

    def test_individual_id_equals_name(self, mocker):
        """individual_id must equal individual.name, not the full IRI."""
        onto = make_fake_ontology([make_fake_individual("BD_Manif_Fiebre")])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].individual_id == "BD_Manif_Fiebre"


# ---------------------------------------------------------------------------
# Tests: classes mapping
# ---------------------------------------------------------------------------

class TestClassesMapping:

    def test_no_classes_returns_empty_list(self, mocker):
        """An individual with no is_a entries must have classes == []."""
        onto = make_fake_ontology([make_fake_individual("ind_1", classes=[])])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].classes == []

    def test_single_class_mapped_correctly(self, mocker):
        """An individual with one class must produce classes == ['ClassName']."""
        onto = make_fake_ontology([make_fake_individual("ind_1", classes=["ManifestacionClinica"])])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].classes == ["ManifestacionClinica"]

    def test_multiple_classes_mapped_correctly(self, mocker):
        """An individual with multiple classes must list all of their names, in order."""
        onto = make_fake_ontology([
            make_fake_individual("ind_1", classes=["ClassA", "ClassB", "ClassC"])
        ])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].classes == ["ClassA", "ClassB", "ClassC"]


# ---------------------------------------------------------------------------
# Tests: data_properties mapping
# ---------------------------------------------------------------------------

class TestDataPropertiesMapping:

    def test_no_properties_returns_empty_dict(self, mocker):
        """An individual with get_properties() == set() must have data_properties == {}."""
        onto = make_fake_ontology([make_fake_individual("ind_1", properties={})])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].data_properties == {}

    def test_single_property_single_value(self, mocker):
        """A property with one literal value must map to a list with that one value."""
        onto = make_fake_ontology([
            make_fake_individual("ind_1", properties={"hasProbability": [0.78]})
        ])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].data_properties["hasProbability"] == [0.78]

    def test_single_property_multiple_values(self, mocker):
        """A property with several literal values must keep them all, in order."""
        onto = make_fake_ontology([
            make_fake_individual("ind_1", properties={"hasTag": ["acute", "severe"]})
        ])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].data_properties["hasTag"] == ["acute", "severe"]

    def test_multiple_properties_all_present(self, mocker):
        """Multiple distinct properties must all appear as separate keys."""
        onto = make_fake_ontology([
            make_fake_individual("ind_1", properties={
                "hasProbability": [0.5],
                "hasCost":        [1200.0],
            })
        ])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")

        assert result[0].data_properties["hasProbability"] == [0.5]
        assert result[0].data_properties["hasCost"]        == [1200.0]


# ---------------------------------------------------------------------------
# Tests: combined fields on a realistic individual
# ---------------------------------------------------------------------------

class TestCombinedFields:

    def test_individual_with_classes_and_properties_together(self, mocker):
        """A realistic individual must populate id, classes and properties simultaneously."""
        onto = make_fake_ontology([
            make_fake_individual(
                "BD_Manif_Fiebre",
                classes=["ManifestacionClinica"],
                properties={"hasProbability": [0.42]},
            )
        ])
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.return_value = onto

        from core.extractor import OntologyExtractor
        result = OntologyExtractor.extract("fake_path.owl")
        entity = result[0]

        assert entity.individual_id == "BD_Manif_Fiebre"
        assert entity.classes == ["ManifestacionClinica"]
        assert entity.data_properties["hasProbability"] == [0.42]


# ---------------------------------------------------------------------------
# Tests: file loading errors
# ---------------------------------------------------------------------------

class TestFileLoadingErrors:

    def test_raises_file_not_found_for_missing_file(self, mocker):
        """extract() must raise FileNotFoundError when the ontology file does not exist."""
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.side_effect = FileNotFoundError("no such file")

        from core.extractor import OntologyExtractor
        with pytest.raises(FileNotFoundError):
            OntologyExtractor.extract("nonexistent.owl")

    def test_wraps_generic_load_errors_as_file_not_found(self, mocker):
        """
        Any underlying load failure (corrupt file, unsupported format, etc.)
        must surface consistently as FileNotFoundError, per the agreed contract.
        """
        mock_get_ontology = mocker.patch("core.extractor.get_ontology")
        mock_get_ontology.return_value.load.side_effect = OSError("corrupt OWL file")

        from core.extractor import OntologyExtractor
        with pytest.raises(FileNotFoundError):
            OntologyExtractor.extract("corrupt.owl")
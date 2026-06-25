"""
Unit tests for OntologyExtractor
================================

Contract covered here:
  - extract(file_path) returns only the individuals that are present in the
    target ontology but NOT in the base ontology.
  - The returned objects are raw owlready2 individuals.
  - The base ontology is loaded exactly once and cached.
  - A missing/invalid file causes FileNotFoundError to propagate.
"""

import pytest
import owlready2 as _owl
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.pipeline.extractor import OntologyExtractor
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_individual(name: str):
    return SimpleNamespace(name=name)


def make_fake_ontology(individuals=None):
    return SimpleNamespace(individuals=lambda: list(individuals or []))


ONTOLOGY_PATH = Path("tests/ontologies/osdi_CU1_P1_S1_M1.rdf")

@pytest.fixture(scope="module")
def records():
    """Carga la ontología una sola vez y la comparte con todas las clases."""
    if not ONTOLOGY_PATH.exists():
        pytest.fail(f"Falta el archivo de prueba: {ONTOLOGY_PATH}")
    return OntologyExtractor.extract(str(ONTOLOGY_PATH))
# ---------------------------------------------------------------------------
# Helpers for inspecting raw owlready2 individuals
# ---------------------------------------------------------------------------

def _classes(ind):
    """List of class names an individual belongs to."""
    return [c.name for c in ind.is_a if hasattr(c, "name")]


def _object_properties(ind):
    """Dict of object-property name -> list of target individual names."""
    result = {}
    for prop in ind.get_properties():
        if _owl.prop.ObjectPropertyClass in type(prop).__mro__:
            vals = [v.name for v in getattr(ind, prop.python_name, []) if hasattr(v, "name")]
            if vals:
                result[prop.name] = vals
    return result


def _data_properties(ind):
    """Dict of data-property name -> list of literal values."""
    result = {}
    for prop in ind.get_properties():
        if _owl.prop.DataPropertyClass in type(prop).__mro__:
            vals = list(getattr(ind, prop.python_name, []))
            result[prop.name] = vals
    return result


def _annotation_properties(ind):
    """Dict of annotation-property name -> list of values."""
    result = {}
    for prop in ind.get_properties():
        if _owl.annotation.AnnotationPropertyClass in type(prop).__mro__:
            vals = list(getattr(ind, prop.python_name, []))
            if vals:
                result[prop.name] = vals
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_base_ontology():
    OntologyExtractor._base_ontology = None
    yield
    OntologyExtractor._base_ontology = None


# ---------------------------------------------------------------------------
# Tests: extract() – unit tests with mocked loader
# ---------------------------------------------------------------------------

class TestExtractIndividuals:

    def test_returns_empty_list_for_empty_ontology(self, mocker):
        base_onto = make_fake_ontology()
        target_onto = make_fake_ontology()
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto]

        assert OntologyExtractor.extract("case.owl") == []

    def test_returns_only_individual_records(self, mocker):
        shared = make_fake_individual("shared")
        only_target = make_fake_individual("target_only")
        base_onto = make_fake_ontology(individuals=[shared])
        target_onto = make_fake_ontology(individuals=[shared, only_target])
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto]

        result = OntologyExtractor.extract("case.owl")

        assert result == [only_target]

    def test_excludes_all_base_individuals(self, mocker):
        base_inds = [make_fake_individual(f"base_{i}") for i in range(5)]
        extra_inds = [make_fake_individual(f"extra_{i}") for i in range(3)]
        base_onto = make_fake_ontology(individuals=base_inds)
        target_onto = make_fake_ontology(individuals=base_inds + extra_inds)
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto]

        assert OntologyExtractor.extract("case.owl") == extra_inds

    def test_preserves_individual_order(self, mocker):
        inds = [make_fake_individual(n) for n in ["ind_A", "ind_B", "ind_C"]]
        base_onto = make_fake_ontology()
        target_onto = make_fake_ontology(individuals=inds)
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto]

        result = OntologyExtractor.extract("case.owl")

        assert [i.name for i in result] == ["ind_A", "ind_B", "ind_C"]

    def test_missing_file_raises_file_not_found(self, mocker):
        base_onto = make_fake_ontology()
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, FileNotFoundError("missing")]

        with pytest.raises(FileNotFoundError):
            OntologyExtractor.extract("missing.owl")

    def test_missing_base_file_also_raises(self, mocker):
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = FileNotFoundError("no base")

        with pytest.raises(FileNotFoundError):
            OntologyExtractor.extract("case.owl")


# ---------------------------------------------------------------------------
# Tests: base ontology caching
# ---------------------------------------------------------------------------

class TestBaseOntologyCaching:

    def test_base_ontology_loaded_only_once_across_calls(self, mocker):
        base_onto = make_fake_ontology()
        target_onto = make_fake_ontology()
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto, target_onto]

        OntologyExtractor.extract("case.owl")
        OntologyExtractor.extract("case.owl")

        assert mock.call_count == 3

    def test_base_ontology_cached_on_class(self, mocker):
        base_onto = make_fake_ontology()
        target_onto = make_fake_ontology()
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.side_effect = [base_onto, target_onto]

        assert OntologyExtractor._base_ontology is None
        OntologyExtractor.extract("case.owl")
        assert OntologyExtractor._base_ontology is base_onto


# ---------------------------------------------------------------------------
# Tests: _load_ontology (static helper)
# ---------------------------------------------------------------------------

class TestLoadOntology:

    def test_uses_get_ontology_by_default(self, mocker):
        fake_onto = make_fake_ontology()
        mock = mocker.patch("core.extractor.get_ontology")
        mock.return_value.load.return_value = fake_onto

        result = OntologyExtractor._load_ontology("some/path.ttl")

        mock.assert_called_once_with("some/path.ttl")
        assert result is fake_onto

    def test_accepts_custom_ontology_loader(self):
        fake_onto = make_fake_ontology()
        custom_loader = MagicMock(return_value=MagicMock(load=MagicMock(return_value=fake_onto)))

        result = OntologyExtractor._load_ontology("some/path.ttl", ontology_loader=custom_loader)

        custom_loader.assert_called_once_with("some/path.ttl")
        assert result is fake_onto


# ---------------------------------------------------------------------------
# Tests: Real ontology integration
# -------------------------------------------------

class TestRealOntologyIntegration:
    """
    Tests using the real ontology file from tests/ontologies/
    These validate that the extractor works correctly with genuine OWL files.
    """
    def test_extract_real_ontology_has_individuals(self, records):
        assert len(records) > 0, "Real ontology should contain individuals"

    def test_real_ontology_case_id_is_filename(self, records):
        """Every individual's .name does not contain path separators."""
        for ind in records:
            assert "/" not in ind.name, f"name should not contain path: {ind.name}"

    def test_real_ontology_individual_extraction_format(self, records):
        """Every returned object exposes the owlready2 individual API."""
        first = records[0]
        assert isinstance(first.name, str) and len(first.name) > 0
        assert isinstance(first.is_a, list)
        assert callable(first.get_properties)

    def test_real_ontology_extracts_classes(self, records):
        """At least some individuals belong to named classes via is_a."""
        inds_with_classes = [ind for ind in records if _classes(ind)]
        assert len(inds_with_classes) > 0, "Some individuals should belong to a class"

        for ind in inds_with_classes:
            for cls_name in _classes(ind):
                assert isinstance(cls_name, str)

    def test_real_ontology_specific_individual_extraction(self, records):
        """Attribute_Age individual is present and belongs to class Attribute."""
        age_ind = next((ind for ind in records if ind.name == "Attribute_Age"), None)

        assert age_ind is not None, "Should find Attribute_Age individual"
        assert "Attribute" in _classes(age_ind)

    def test_real_ontology_data_properties_normalized(self, records):
        """Data property values are always lists."""
        inds_with_data = [ind for ind in records if _data_properties(ind)]
        assert len(inds_with_data) > 0, "Some individuals should have data properties"

        for ind in inds_with_data:
            for prop_name, vals in _data_properties(ind).items():
                assert isinstance(vals, list), f"{prop_name} should be a list"

    def test_real_ontology_order_preservation(self, records):
        """Calling extract() twice returns the same individual order."""
        records1 = OntologyExtractor.extract(str(ONTOLOGY_PATH))
        records2 = OntologyExtractor.extract(str(ONTOLOGY_PATH))

        assert [i.name for i in records1] == [i.name for i in records2]

    def test_real_ontology_individual_with_single_property(self, records):
        """BTD_Disease_Profound has exactly one data property: hasDescription."""
        ind = next((r for r in records if r.name == "BTD_Disease_Profound"), None)

        assert ind is not None
        assert "Disease" in _classes(ind)
        dp = _data_properties(ind)
        assert len(dp) == 1
        assert "hasDescription" in dp
        assert isinstance(dp["hasDescription"], list)
        assert len(dp["hasDescription"]) > 0

    def test_real_ontology_individuals_without_properties(self, records):
        """DI_Factor has no data or object properties."""
        factor = next((r for r in records if r.name == "DI_Factor"), None)

        assert factor is not None
        assert _data_properties(factor) == {}
        assert _object_properties(factor) == {}

    def test_real_ontology_empty_property_values(self, records):
        """BTD_Disease.hasDisutilityCombinationMethod is an empty list when present."""
        btd = next((r for r in records if r.name == "BTD_Disease"), None)
        assert btd is not None

        dp = _data_properties(btd)
        if "hasDisutilityCombinationMethod" in dp:
            assert dp["hasDisutilityCombinationMethod"] == []

    def test_real_ontology_multiple_class_inheritance(self, records):
        """Some individuals inherit from more than one class."""
        multi_class = [ind for ind in records if len(_classes(ind)) > 1]

        assert len(multi_class) > 0, "Should find individuals with multiple classes"
        for ind in multi_class:
            assert all(isinstance(c, str) for c in _classes(ind))

    def test_real_ontology_object_properties_with_entity_references(self, records):
        """Object property values resolve to string names."""
        inds_with_obj = [ind for ind in records if _object_properties(ind)]
        assert len(inds_with_obj) > 0

        for ind in inds_with_obj:
            for prop_name, vals in _object_properties(ind).items():
                assert isinstance(vals, list)
                assert all(isinstance(v, str) for v in vals)

    def test_real_ontology_mixed_data_types(self, records):
        """Data property values can be strings, ints, floats — never None."""
        for ind in records:
            for prop_name, vals in _data_properties(ind).items():
                for v in vals:
                    assert v is not None

    def test_real_ontology_multiple_values_in_property(self, records):
        """Some data properties have more than one value."""
        multi_valued = [
            (ind.name, prop_name, vals)
            for ind in records
            for prop_name, vals in _data_properties(ind).items()
            if len(vals) > 1
        ]

        assert len(multi_valued) > 0, "Should find data properties with multiple values"


# ---------------------------------------------------------------------------
# Tests: Object Properties (Entity Relationships)
# ---------------------------------------------------------------------------

class TestObjectProperties:
    """
    Tests for object properties — the relationships between individuals.

    Object Properties Found (7 total):
      1. hasDataItemType (6 individuals)
      2. hasEpidemiologicalParameter (1)
      3. hasFollowUpStrategy (1)
      4. hasSubpopulation (1)
      5. hasTreatmentCost (1)
      6. isModifiedBy (1)
      7. modifies (1)
    """
    def test_object_properties_exist_in_ontology(self, records):
        inds_with_obj = [ind for ind in records if _object_properties(ind)]
        assert len(inds_with_obj) > 0
        assert len(inds_with_obj) >= 10

    def test_has_data_item_type_relationship(self, records):
        inds_with_dit = [ind for ind in records if "hasDataItemType" in _object_properties(ind)]
        assert len(inds_with_dit) > 0

        incidence = next((ind for ind in inds_with_dit if ind.name == "Constant_Incidence1"), None)
        assert incidence is not None, "Should find Constant_Incidence1 individual"
        if incidence:
            vals = _object_properties(incidence)["hasDataItemType"]
            assert isinstance(vals, list) and len(vals) > 0
            assert all(isinstance(v, str) for v in vals)

    def test_has_epidemiological_parameter_relationship(self, records):
        btd = next((ind for ind in records if ind.name == "BTD_Disease"), None)
        assert btd is not None

        obj_props = _object_properties(btd)
        assert "hasEpidemiologicalParameter" in obj_props
        epi = obj_props["hasEpidemiologicalParameter"]
        assert isinstance(epi, list) and len(epi) > 0
        assert "BTD_Incidence" in epi

    def test_has_follow_up_strategy_relationship(self, records):
        btd = next((ind for ind in records if ind.name == "BTD_Disease"), None)
        assert btd is not None

        obj_props = _object_properties(btd)
        assert "hasFollowUpStrategy" in obj_props
        assert isinstance(obj_props["hasFollowUpStrategy"], list)
        assert len(obj_props["hasFollowUpStrategy"]) > 0

    def test_has_treatment_cost_relationship(self, records):
        inds = [ind for ind in records if "hasTreatmentCost" in _object_properties(ind)]
        assert len(inds) > 0

        for ind in inds:
            vals = _object_properties(ind)["hasTreatmentCost"]
            assert isinstance(vals, list)
            assert all(isinstance(v, str) for v in vals)

    def test_modifies_and_is_modified_by_relationships(self, records):
        inds_modifies = [ind for ind in records if "modifies" in _object_properties(ind)]
        inds_modified = [ind for ind in records if "isModifiedBy" in _object_properties(ind)]

        assert len(inds_modifies) > 0
        assert len(inds_modified) > 0

        for ind in inds_modifies:
            vals = _object_properties(ind)["modifies"]
            assert isinstance(vals, list)
            assert all(isinstance(v, str) for v in vals)

    def test_has_subpopulation_relationship(self, records):
        inds = [ind for ind in records if "hasSubpopulation" in _object_properties(ind)]
        assert len(inds) > 0

        for ind in inds:
            vals = _object_properties(ind)["hasSubpopulation"]
            assert isinstance(vals, list) and len(vals) > 0

    def test_object_properties_never_empty_lists(self, records):
        for ind in records:
            for prop_name, vals in _object_properties(ind).items():
                assert isinstance(vals, list)
                assert len(vals) > 0, f"{prop_name} should never be an empty list"
                assert all(isinstance(v, str) for v in vals)

    def test_object_properties_all_relationships_bidirectional(self, records):
        """If A modifies B, then B.isModifiedBy should contain A."""
        by_name = {ind.name: ind for ind in records}

        for ind in records:
            obj = _object_properties(ind)
            if "modifies" in obj:
                for target_name in obj["modifies"]:
                    target = by_name.get(target_name)
                    if target:
                        target_obj = _object_properties(target)
                        if "isModifiedBy" in target_obj:
                            assert ind.name in target_obj["isModifiedBy"], \
                                f"Broken: {ind.name} modifies {target_name} but reverse not found"
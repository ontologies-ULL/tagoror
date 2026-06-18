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


# ---------------------------------------------------------------------------
# Tests: Real ontology integration
# ---------------------------------------------------------------------------

class TestRealOntologyIntegration:
    """
    Tests using the real ontology file from tests/ontologies/
    These validate that the extractor works correctly with genuine OWL files.
    """

    def test_extract_real_ontology_has_individuals(self):
        """
        Verify the real ontology can be loaded and produces extraction records.
        """
        from core.extractor import OntologyExtractor
        from pathlib import Path

        ontology_path = Path("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        assert ontology_path.exists(), "Real ontology file should exist"

        records = OntologyExtractor.extract(str(ontology_path))

        assert len(records) > 0, "Real ontology should contain individuals"
        assert len(records) == 71, "Expected 71 individuals in this ontology"

    def test_real_ontology_case_id_is_filename(self):
        """
        Verify CaseID is set to filename only, not full path.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # All records should have same CaseID = filename
        for record in records:
            assert record.CaseID == "osdi_CU1_P1_S1_M1.owx"

    def test_real_ontology_individual_extraction_format(self):
        """
        Verify that extracted records have the correct structure and format.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Check first individual as a sample
        first = records[0]
        
        # Verify all fields exist and are correct types
        assert isinstance(first.CaseID, str)
        assert isinstance(first.Individual, str)
        assert isinstance(first.Classes, list)
        assert isinstance(first.ObjectProperties, dict)
        assert isinstance(first.DataProperties, dict)
        assert isinstance(first.Annotations, dict)
        
        # Verify Individual field is populated
        assert len(first.Individual) > 0

    def test_real_ontology_extracts_classes(self):
        """
        Verify that ontology classes are correctly extracted.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find records with classes
        records_with_classes = [r for r in records if r.Classes]
        assert len(records_with_classes) > 0, "Some individuals should have classes"
        
        # Check structure: all Classes should be strings
        for record in records_with_classes:
            for cls_name in record.Classes:
                assert isinstance(cls_name, str)

    def test_real_ontology_specific_individual_extraction(self):
        """
        Test extraction of a known individual with expected properties.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find Attribute_Age individual
        age_record = next(
            (r for r in records if r.Individual == "Attribute_Age"),
            None
        )
        
        assert age_record is not None, "Should find Attribute_Age individual"
        assert age_record.CaseID == "osdi_CU1_P1_S1_M1.owx"
        assert age_record.Individual == "Attribute_Age"
        assert "Attribute" in age_record.Classes

    def test_real_ontology_data_properties_normalized(self):
        """
        Verify that data properties are correctly normalized to lists.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find an individual with data properties
        records_with_data_props = [r for r in records if r.DataProperties]
        assert len(records_with_data_props) > 0, "Some individuals should have data properties"
        
        # Verify all data property values are lists
        for record in records_with_data_props:
            for prop_name, prop_values in record.DataProperties.items():
                assert isinstance(prop_values, list), f"{prop_name} values should be a list"

    def test_real_ontology_order_preservation(self):
        """
        Verify that individual extraction order is consistent.
        """
        from core.extractor import OntologyExtractor

        records1 = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        records2 = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        names1 = [r.Individual for r in records1]
        names2 = [r.Individual for r in records2]
        
        assert names1 == names2, "Individual extraction order should be consistent"

    def test_real_ontology_individual_with_single_property(self):
        """
        Test extraction of individuals with minimal properties (like BTD_Disease_Profound).
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find BTD_Disease_Profound - a disease with only hasDescription
        profound_record = next(
            (r for r in records if r.Individual == "BTD_Disease_Profound"),
            None
        )
        
        assert profound_record is not None
        assert profound_record.Individual == "BTD_Disease_Profound"
        assert "Disease" in profound_record.Classes
        # Only has hasDescription
        assert len(profound_record.DataProperties) == 1
        assert "hasDescription" in profound_record.DataProperties
        assert isinstance(profound_record.DataProperties["hasDescription"], list)
        assert len(profound_record.DataProperties["hasDescription"]) > 0

    def test_real_ontology_individuals_without_properties(self):
        """
        Test extraction of individuals with NO properties at all.
        Examples: DI_Factor, DI_MeanDifference
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals without any properties
        empty_property_records = [
            r for r in records
            if not r.DataProperties and not r.ObjectProperties
        ]
        
        assert len(empty_property_records) > 0, "Should find individuals without properties"
        # Verify at least DI_Factor is there
        factor_record = next(
            (r for r in empty_property_records if r.Individual == "DI_Factor"),
            None
        )
        assert factor_record is not None
        assert factor_record.DataProperties == {}
        assert factor_record.ObjectProperties == {}

    def test_real_ontology_empty_property_values(self):
        """
        Test extraction handles empty property values correctly.
        Example: BTD_Disease.hasDisutilityCombinationMethod is empty list
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find BTD_Disease which has an empty property
        btd_disease_record = next(
            (r for r in records if r.Individual == "BTD_Disease"),
            None
        )
        
        assert btd_disease_record is not None
        # BTD_Disease should have hasDisutilityCombinationMethod as empty
        if "hasDisutilityCombinationMethod" in btd_disease_record.DataProperties:
            assert btd_disease_record.DataProperties["hasDisutilityCombinationMethod"] == []

    def test_real_ontology_multiple_class_inheritance(self):
        """
        Test extraction of individuals with multiple class inheritance.
        Example: Constant_Incidence1 inherits from ['DeterministicParameter', 'EpidemiologicalParameter']
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with multiple classes
        multi_class_records = [r for r in records if len(r.Classes) > 1]
        
        assert len(multi_class_records) > 0, "Should find individuals with multiple classes"
        
        # Check structure
        for record in multi_class_records:
            assert isinstance(record.Classes, list)
            assert len(record.Classes) > 1
            # All class names should be strings
            for cls_name in record.Classes:
                assert isinstance(cls_name, str)

    def test_real_ontology_object_properties_with_entity_references(self):
        """
        Test that object properties correctly resolve entity references.
        Example: hasFollowUpStrategy, hasEpidemiologicalParameter
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with object properties
        with_object_props = [
            r for r in records 
            if r.ObjectProperties
        ]
        
        assert len(with_object_props) > 0, "Should find individuals with object properties"
        
        # Verify object property values are properly resolved
        for record in with_object_props:
            for prop_name, prop_values in record.ObjectProperties.items():
                assert isinstance(prop_values, list)
                # Values should be strings (entity names)
                for value in prop_values:
                    assert isinstance(value, str)

    def test_real_ontology_mixed_data_types(self):
        """
        Test extraction handles mixed data types correctly.
        Examples: strings, numbers (floats, ints), etc.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find records with data properties
        for record in records:
            if record.DataProperties:
                for prop_name, prop_values in record.DataProperties.items():
                    assert isinstance(prop_values, list)
                    for value in prop_values:
                        # Values can be str, int, float, bool, etc.
                        assert value is not None

    def test_real_ontology_multiple_values_in_property(self):
        """
        Test extraction of properties with multiple values.
        Example: hasExpectedValue or hasSource with 2+ items
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find records with multi-valued properties
        multi_valued_records = [
            (r.Individual, prop_name, len(values))
            for r in records
            for prop_name, values in r.DataProperties.items()
            if len(values) > 1
        ]
        
        # Verify we found some multi-valued properties
        assert len(multi_valued_records) > 0, "Should find properties with multiple values"
        
        # Verify the structure is correct
        for individual_name, prop_name, value_count in multi_valued_records[:3]:
            record = next(r for r in records if r.Individual == individual_name)
            assert len(record.DataProperties[prop_name]) == value_count


# ---------------------------------------------------------------------------
# Tests: Object Properties (Entity Relationships)
# ---------------------------------------------------------------------------

class TestObjectProperties:
    """
    Tests for object properties - the relationships between individuals.
    These are key for understanding domain relationships in the ontology.
    
    Object Properties Found (7 total):
      1. hasDataItemType (6 individuals)
      2. hasEpidemiologicalParameter (1)
      3. hasFollowUpStrategy (1)
      4. hasSubpopulation (1)
      5. hasTreatmentCost (1)
      6. isModifiedBy (1)
      7. modifies (1)
    """

    def test_object_properties_exist_in_ontology(self):
        """
        Verify that object properties are correctly identified and extracted.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find all records with any object properties
        records_with_object_props = [r for r in records if r.ObjectProperties]
        
        assert len(records_with_object_props) > 0, "Should find records with object properties"
        # We know there should be at least some records with object properties
        assert len(records_with_object_props) >= 10, "Should have multiple individuals with relationships"

    def test_has_data_item_type_relationship(self):
        """
        Test extraction of hasDataItemType relationships.
        This maps constants to their data item types (e.g., Incidence, Prevalence).
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with hasDataItemType
        records_with_data_item_type = [
            r for r in records 
            if "hasDataItemType" in r.ObjectProperties
        ]
        
        assert len(records_with_data_item_type) > 0, "Should find hasDataItemType relationships"
        
        # Check specific known examples
        incidence_record = next(
            (r for r in records_with_data_item_type 
             if r.Individual == "Constant_Incidence1"),
            None
        )
        
        if incidence_record:
            assert "hasDataItemType" in incidence_record.ObjectProperties
            data_item_types = incidence_record.ObjectProperties["hasDataItemType"]
            assert isinstance(data_item_types, list)
            assert len(data_item_types) > 0
            # Should resolve to entity names
            assert all(isinstance(name, str) for name in data_item_types)

    def test_has_epidemiological_parameter_relationship(self):
        """
        Test extraction of hasEpidemiologicalParameter relationships.
        This links diseases to their epidemiological parameters.
        Example: BTD_Disease -> BTD_Incidence
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find BTD_Disease
        btd_disease = next(
            (r for r in records if r.Individual == "BTD_Disease"),
            None
        )
        
        assert btd_disease is not None, "Should find BTD_Disease"
        assert "hasEpidemiologicalParameter" in btd_disease.ObjectProperties
        
        # Verify the relationship
        epi_params = btd_disease.ObjectProperties["hasEpidemiologicalParameter"]
        assert isinstance(epi_params, list)
        assert len(epi_params) > 0
        # Should have BTD_Incidence
        assert "BTD_Incidence" in epi_params

    def test_has_follow_up_strategy_relationship(self):
        """
        Test extraction of hasFollowUpStrategy relationships.
        This links diseases to their follow-up strategies.
        Example: BTD_Disease -> BTD_FollowUp_AnnualMonitoring
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find BTD_Disease
        btd_disease = next(
            (r for r in records if r.Individual == "BTD_Disease"),
            None
        )
        
        assert btd_disease is not None
        assert "hasFollowUpStrategy" in btd_disease.ObjectProperties
        
        # Verify the relationship
        follow_up = btd_disease.ObjectProperties["hasFollowUpStrategy"]
        assert isinstance(follow_up, list)
        assert len(follow_up) > 0

    def test_has_treatment_cost_relationship(self):
        """
        Test extraction of hasTreatmentCost relationships.
        This links interventions to their cost data items.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with hasTreatmentCost
        records_with_treatment_cost = [
            r for r in records 
            if "hasTreatmentCost" in r.ObjectProperties
        ]
        
        assert len(records_with_treatment_cost) > 0, "Should find hasTreatmentCost relationships"
        
        # Verify structure
        for record in records_with_treatment_cost:
            treatment_costs = record.ObjectProperties["hasTreatmentCost"]
            assert isinstance(treatment_costs, list)
            # Values should be resolved entity names
            assert all(isinstance(name, str) for name in treatment_costs)

    def test_modifies_and_is_modified_by_relationships(self):
        """
        Test extraction of modifies and isModifiedBy relationships.
        These are inverse relationships (bidirectional) representing interventions
        that modify disease manifestations.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with modifies
        records_with_modifies = [
            r for r in records 
            if "modifies" in r.ObjectProperties
        ]
        
        # Find individuals with isModifiedBy
        records_with_is_modified = [
            r for r in records 
            if "isModifiedBy" in r.ObjectProperties
        ]
        
        # Should find at least one of each
        assert len(records_with_modifies) > 0, "Should find modifies relationships"
        assert len(records_with_is_modified) > 0, "Should find isModifiedBy relationships"
        
        # Verify structure
        for record in records_with_modifies:
            modifies_values = record.ObjectProperties["modifies"]
            assert isinstance(modifies_values, list)
            assert all(isinstance(name, str) for name in modifies_values)

    def test_has_subpopulation_relationship(self):
        """
        Test extraction of hasSubpopulation relationships.
        This links populations to their subpopulations.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Find individuals with hasSubpopulation
        records_with_subpopulation = [
            r for r in records 
            if "hasSubpopulation" in r.ObjectProperties
        ]
        
        assert len(records_with_subpopulation) > 0, "Should find hasSubpopulation relationships"
        
        # Verify structure
        for record in records_with_subpopulation:
            subpop = record.ObjectProperties["hasSubpopulation"]
            assert isinstance(subpop, list)
            assert len(subpop) > 0

    def test_object_properties_never_empty_lists(self):
        """
        Verify that when an object property is included, it always has values.
        Empty lists in ObjectProperties should not exist.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Check all records
        for record in records:
            for prop_name, prop_values in record.ObjectProperties.items():
                assert isinstance(prop_values, list), f"{prop_name} should be a list"
                assert len(prop_values) > 0, f"{prop_name} should never have empty list (strip it instead)"
                # All values should be strings (resolved names)
                assert all(isinstance(v, str) for v in prop_values), f"{prop_name} values should all be strings"

    def test_object_properties_all_relationships_bidirectional(self):
        """
        Test that bidirectional relationships (modifies <-> isModifiedBy) are consistent.
        If A modifies B, then B should be isModifiedBy A.
        """
        from core.extractor import OntologyExtractor

        records = OntologyExtractor.extract("tests/ontologies/osdi_CU1_P1_S1_M1.owx")
        
        # Create lookup by individual name
        by_name = {r.Individual: r for r in records}
        
        # Check bidirectional relationships
        for record in records:
            if "modifies" in record.ObjectProperties:
                # A modifies B
                for target_name in record.ObjectProperties["modifies"]:
                    # B should exist
                    target_record = by_name.get(target_name)
                    if target_record:
                        # B isModifiedBy A
                        if "isModifiedBy" in target_record.ObjectProperties:
                            assert record.Individual in target_record.ObjectProperties["isModifiedBy"], \
                                f"Bidirectional relationship broken: {record.Individual} modifies {target_name} but reverse not found"

"""
Unit tests for TurtleSerializer
==================================
Based on: turtle_serializer.py (owlready2 + rdflib + OpenTelemetry)

Covers:
  - __init__: tracer stored on the instance
  - process_ontology (normal):
      * returns the string produced by graph.serialize
      * default_world.as_rdflib_graph called with the ontology
      * graph.serialize called with format="turtle"
      * span attributes: ontology_iri, ontology_name
      * span status OK on success
  - process_ontology (extreme):
      * exception from as_rdflib_graph propagates unchanged, span ERROR + description
      * exception from graph.serialize propagates unchanged, span ERROR + description
      * empty/falsy base_iri and name are still set as span attributes
      * empty serialized result ("") is returned as-is, no exception
  - process_individual (normal):
      * returns the string produced by individual_graph.serialize
      * URIRef built from individual.iri
      * default_world.as_rdflib_graph called with NO arguments (whole world graph)
      * only triples matching the subject are added to individual_graph
      * individual_graph.serialize called with format="turtle", subject, graph
      * span attributes: individual_iri, individual_name
      * span status OK on success
  - process_individual (extreme):
      * exception constructing URIRef propagates unchanged, span ERROR + description
      * exception from full_graph.triples propagates unchanged, span ERROR + description
      * exception from individual_graph.serialize propagates unchanged, span ERROR + description
      * individual with zero matching triples: serialize is still called, add() never called
      * individual with a large number of matching triples: all of them are added, none dropped
      * non-ASCII individual name/iri flow through untouched

Patching strategy:
  - default_world, Graph and URIRef are patched at their import site inside
    turtle_serializer.py, since owlready2/rdflib are not mocked at the
    library level — only the specific symbols the module uses.
  - The tracer is a plain MagicMock configured to behave like a context
    manager (start_as_current_span(...).__enter__ returns the span mock).

Note: unlike GeminiClient, TurtleSerializer's error path does NOT call
span.record_exception — it only sets Status(ERROR, description=str(error))
and re-raises. Tests reflect that; asserting record_exception here would be
testing behavior that doesn't exist in the class.
"""

import pytest
from unittest.mock import MagicMock
from opentelemetry.trace import StatusCode


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_span():
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


@pytest.fixture
def mock_tracer(mock_span):
    tracer = MagicMock()
    tracer.start_as_current_span.return_value = mock_span
    return tracer


@pytest.fixture
def serializer(mock_tracer):
    from serialization.serializers.turtle_serializer import TurtleSerializer
    return TurtleSerializer(tracer=mock_tracer)


@pytest.fixture
def mock_ontology():
    ontology = MagicMock()
    ontology.base_iri = "http://example.org/onto#"
    ontology.name = "test_ontology"
    return ontology


@pytest.fixture
def mock_individual():
    individual = MagicMock()
    individual.iri = "http://example.org/onto#MyIndividual"
    individual.name = "MyIndividual"
    return individual


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_tracer_is_stored(self, mock_tracer):
        from serialization.serializers.turtle_serializer import TurtleSerializer
        instance = TurtleSerializer(tracer=mock_tracer)
        assert instance.tracer is mock_tracer


# ---------------------------------------------------------------------------
# Tests: process_ontology — normal behavior
# ---------------------------------------------------------------------------

class TestProcessOntology:

    def test_returns_serialized_turtle_string(self, serializer, mock_ontology, mocker):
        mock_graph = MagicMock()
        mock_graph.serialize.return_value = "@prefix ex: <http://example.org/> .\nex:Thing a owl:Class ."
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_graph,
        )
        result = serializer.process_ontology(mock_ontology)
        assert result == mock_graph.serialize.return_value

    def test_as_rdflib_graph_called_with_ontology(self, serializer, mock_ontology, mocker):
        mock_as_rdflib = mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(),
        )
        serializer.process_ontology(mock_ontology)
        mock_as_rdflib.assert_called_once_with(mock_ontology)

    def test_graph_serialize_called_with_turtle_format(self, serializer, mock_ontology, mocker):
        mock_graph = MagicMock()
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_graph,
        )
        serializer.process_ontology(mock_ontology)
        mock_graph.serialize.assert_called_once_with(format="turtle")

    def test_span_attributes_set(self, serializer, mock_ontology, mock_span, mocker):
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(),
        )
        serializer.process_ontology(mock_ontology)
        attr_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["ontology_iri"] == mock_ontology.base_iri
        assert attr_calls["ontology_name"] == mock_ontology.name

    def test_span_status_set_to_ok(self, serializer, mock_ontology, mock_span, mocker):
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(),
        )
        serializer.process_ontology(mock_ontology)
        status_arg = mock_span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Tests: process_ontology — extreme / edge cases
# ---------------------------------------------------------------------------

class TestProcessOntologyExtreme:

    def test_reraises_exception_from_as_rdflib_graph(self, serializer, mock_ontology, mocker):
        """If fetching the rdflib graph blows up, the original exception must propagate."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            side_effect=RuntimeError("owlready2 world is corrupted"),
        )
        with pytest.raises(RuntimeError, match="owlready2 world is corrupted"):
            serializer.process_ontology(mock_ontology)

    def test_reraises_exception_from_graph_serialize(self, serializer, mock_ontology, mocker):
        """If graph.serialize blows up, the original exception must propagate."""
        mock_graph = MagicMock()
        mock_graph.serialize.side_effect = ValueError("unsupported rdflib plugin")
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_graph,
        )
        with pytest.raises(ValueError, match="unsupported rdflib plugin"):
            serializer.process_ontology(mock_ontology)

    def test_span_status_error_with_description_on_failure(self, serializer, mock_ontology, mock_span, mocker):
        """Span status must be ERROR and description must contain the error message."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(RuntimeError):
            serializer.process_ontology(mock_ontology)
        status_arg = mock_span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR
        assert "boom" in status_arg.description

    def test_empty_base_iri_and_name_still_set_as_attributes(self, serializer, mock_span, mocker):
        """Falsy metadata (empty strings) must still be forwarded as span attributes,
        not silently skipped."""
        ontology = MagicMock()
        ontology.base_iri = ""
        ontology.name = ""
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(),
        )
        serializer.process_ontology(ontology)
        attr_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["ontology_iri"] == ""
        assert attr_calls["ontology_name"] == ""

    def test_empty_serialized_result_returned_as_is(self, serializer, mock_ontology, mocker):
        """An ontology that serializes to an empty string is not an error condition."""
        mock_graph = MagicMock()
        mock_graph.serialize.return_value = ""
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_graph,
        )
        result = serializer.process_ontology(mock_ontology)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: process_individual — normal behavior
# ---------------------------------------------------------------------------

class TestProcessIndividual:

    def _patch_common(self, mocker, triples=None):
        """
        Helper to patch URIRef, default_world.as_rdflib_graph (no-arg call) and
        the Graph() constructor used for the individual-scoped graph.
        Returns (mock_uriref_cls, mock_full_graph, mock_graph_cls, mock_individual_graph).
        """
        mock_subject_uriref = MagicMock(name="subject_uriref")
        mock_uriref_cls = mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            return_value=mock_subject_uriref,
        )

        mock_full_graph = MagicMock()
        mock_full_graph.triples.return_value = triples if triples is not None else []
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_full_graph,
        )

        mock_individual_graph = MagicMock()
        mock_individual_graph.serialize.return_value = "@prefix ex: <http://example.org/> .\nex:MyIndividual a owl:Thing ."
        mock_graph_cls = mocker.patch(
            "serialization.serializers.turtle_serializer.Graph",
            return_value=mock_individual_graph,
        )

        return mock_uriref_cls, mock_full_graph, mock_graph_cls, mock_individual_graph, mock_subject_uriref

    def test_returns_serialized_turtle_string(self, serializer, mock_individual, mocker):
        *_, mock_individual_graph, _ = self._patch_common(mocker)
        result = serializer.process_individual(mock_individual)
        assert result == mock_individual_graph.serialize.return_value

    def test_uriref_constructed_from_individual_iri(self, serializer, mock_individual, mocker):
        mock_uriref_cls, *_ = self._patch_common(mocker)
        serializer.process_individual(mock_individual)
        mock_uriref_cls.assert_called_once_with(mock_individual.iri)

    def test_as_rdflib_graph_called_with_no_arguments(self, serializer, mock_individual, mocker):
        """process_individual must fetch the WHOLE world graph (no ontology filter)."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            return_value=MagicMock(),
        )
        mock_as_rdflib = mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(triples=MagicMock(return_value=[])),
        )
        mocker.patch(
            "serialization.serializers.turtle_serializer.Graph",
            return_value=MagicMock(),
        )
        serializer.process_individual(mock_individual)
        mock_as_rdflib.assert_called_once_with()

    def test_matching_triples_are_added_to_individual_graph(self, serializer, mock_individual, mocker):
        triple_1 = ("s1", "p1", "o1")
        triple_2 = ("s1", "p2", "o2")
        *_, mock_individual_graph, _ = self._patch_common(mocker, triples=[triple_1, triple_2])
        serializer.process_individual(mock_individual)
        mock_individual_graph.add.assert_any_call(triple_1)
        mock_individual_graph.add.assert_any_call(triple_2)
        assert mock_individual_graph.add.call_count == 2

    def test_serialize_called_with_correct_arguments(self, serializer, mock_individual, mocker):
        _, mock_full_graph, _, mock_individual_graph, mock_subject_uriref = self._patch_common(mocker)
        serializer.process_individual(mock_individual)
        mock_individual_graph.serialize.assert_called_once_with(
            format="turtle", subject=mock_subject_uriref, graph=mock_full_graph
        )

    def test_span_attributes_set(self, serializer, mock_individual, mock_span, mocker):
        self._patch_common(mocker)
        serializer.process_individual(mock_individual)
        attr_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["individual_iri"] == mock_individual.iri
        assert attr_calls["individual_name"] == mock_individual.name

    def test_span_status_set_to_ok(self, serializer, mock_individual, mock_span, mocker):
        self._patch_common(mocker)
        serializer.process_individual(mock_individual)
        status_arg = mock_span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Tests: process_individual — extreme / edge cases
# ---------------------------------------------------------------------------

class TestProcessIndividualExtreme:

    def test_reraises_exception_when_uriref_construction_fails(self, serializer, mock_individual, mocker):
        """A malformed IRI causing URIRef(...) to raise must propagate unchanged."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            side_effect=ValueError("Invalid IRI syntax"),
        )
        with pytest.raises(ValueError, match="Invalid IRI syntax"):
            serializer.process_individual(mock_individual)

    def test_reraises_exception_from_full_graph_triples(self, serializer, mock_individual, mocker):
        """If iterating the world graph's triples blows up, propagate unchanged."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            return_value=MagicMock(),
        )
        mock_full_graph = MagicMock()
        mock_full_graph.triples.side_effect = RuntimeError("graph store closed")
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_full_graph,
        )
        with pytest.raises(RuntimeError, match="graph store closed"):
            serializer.process_individual(mock_individual)

    def test_reraises_exception_from_individual_graph_serialize(self, serializer, mock_individual, mocker):
        """If the final serialize() call blows up, propagate unchanged."""
        mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            return_value=MagicMock(),
        )
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=MagicMock(triples=MagicMock(return_value=[])),
        )
        mock_individual_graph = MagicMock()
        mock_individual_graph.serialize.side_effect = ValueError("cannot serialize disconnected node")
        mocker.patch(
            "serialization.serializers.turtle_serializer.Graph",
            return_value=mock_individual_graph,
        )
        with pytest.raises(ValueError, match="cannot serialize disconnected node"):
            serializer.process_individual(mock_individual)

    def test_span_status_error_with_description_on_failure(self, serializer, mock_individual, mock_span, mocker):
        mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(RuntimeError):
            serializer.process_individual(mock_individual)
        status_arg = mock_span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR
        assert "boom" in status_arg.description

    def test_zero_matching_triples_still_calls_serialize(self, serializer, mock_individual, mocker):
        """An orphan individual with no triples in the world graph must still
        produce a (likely empty/minimal) serialization instead of failing."""
        *_, mock_individual_graph, _ = self._patch_common_static(mocker, triples=[])
        result = serializer.process_individual(mock_individual)
        mock_individual_graph.add.assert_not_called()
        mock_individual_graph.serialize.assert_called_once()
        assert result == mock_individual_graph.serialize.return_value

    def test_large_number_of_matching_triples_all_added(self, serializer, mock_individual, mocker):
        """No triples should be silently dropped, even with a large fan-out."""
        many_triples = [(f"s{i}", f"p{i}", f"o{i}") for i in range(500)]
        *_, mock_individual_graph, _ = self._patch_common_static(mocker, triples=many_triples)
        serializer.process_individual(mock_individual)
        assert mock_individual_graph.add.call_count == 500
        for triple in many_triples:
            mock_individual_graph.add.assert_any_call(triple)

    def test_non_ascii_individual_name_and_iri_are_forwarded_untouched(self, serializer, mock_span, mocker):
        """Unicode IRIs/names must flow through to span attributes and URIRef
        construction without transformation or crashes."""
        individual = MagicMock()
        individual.iri = "http://example.org/onto#Paciente_Ñoño_日本語"
        individual.name = "Paciente_Ñoño_日本語"

        mock_uriref_cls, *_ = self._patch_common_static(mocker, triples=[])
        serializer.process_individual(individual)

        mock_uriref_cls.assert_called_once_with(individual.iri)
        attr_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["individual_iri"] == individual.iri
        assert attr_calls["individual_name"] == individual.name

    # -- shared helper duplicated as a static-friendly version for this class --
    @staticmethod
    def _patch_common_static(mocker, triples=None):
        mock_subject_uriref = MagicMock(name="subject_uriref")
        mock_uriref_cls = mocker.patch(
            "serialization.serializers.turtle_serializer.URIRef",
            return_value=mock_subject_uriref,
        )

        mock_full_graph = MagicMock()
        mock_full_graph.triples.return_value = triples if triples is not None else []
        mocker.patch(
            "serialization.serializers.turtle_serializer.default_world.as_rdflib_graph",
            return_value=mock_full_graph,
        )

        mock_individual_graph = MagicMock()
        mock_individual_graph.serialize.return_value = "@prefix ex: <http://example.org/> .\nex:Individual a owl:Thing ."
        mock_graph_cls = mocker.patch(
            "serialization.serializers.turtle_serializer.Graph",
            return_value=mock_individual_graph,
        )

        return mock_uriref_cls, mock_full_graph, mock_graph_cls, mock_individual_graph, mock_subject_uriref
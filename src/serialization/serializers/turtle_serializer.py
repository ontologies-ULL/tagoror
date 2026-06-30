from owlready2 import Thing, Ontology, default_world
from serialization.base_serializer import BaseSerializer
from rdflib import Graph, URIRef
from opentelemetry.trace import Tracer, Status, StatusCode

class TurtleSerializer(BaseSerializer):
    """
    Concrete implementation of BaseSerializer that serializes OWL individuals and ontologies into Turtle format.
    """

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer

    def process_ontology(self, ontology: Ontology) -> str:
        """
        Serializes an OWL ontology into a Turtle string.

        Args:
            ontology: The OWL ontology to serialize.

        Returns:
            A Turtle string representation of the ontology.
        """
        with self.tracer.start_as_current_span("serialize_ontology_to_turtle") as span:
            span.set_attribute("ontology_iri", ontology.base_iri)
            span.set_attribute("ontology_name", ontology.name)
            try:
                graph = default_world.as_rdflib_graph(ontology)
                result = graph.serialize(format="turtle")

                span.set_status(Status(status_code=StatusCode.OK))
                return result
            except Exception as error:
                span.set_status(Status(status_code=StatusCode.ERROR, description=str(error)))
                
                raise

    def process_individual(self, individual: Thing) -> str:
        """
        Serializes an OWL individual into a Turtle string.

        Args:
            individual: The OWL individual to serialize.

        Returns:
            A Turtle string representation of the individual.
        """
        with self.tracer.start_as_current_span("serialize_individual_to_turtle") as span:
            span.set_attribute("individual_iri", individual.iri)
            span.set_attribute("individual_name", individual.name)
            try:
                subject_iri = URIRef(individual.iri)
                full_graph = default_world.as_rdflib_graph()
                individual_graph = Graph()
                
                for entity, relationship, target in full_graph.triples((subject_iri, None, None)):
                    individual_graph.add((entity, relationship, target))
                
                result = individual_graph.serialize(format="turtle", subject=subject_iri, graph=full_graph)
                span.set_status(Status(status_code=StatusCode.OK))
                return result
            except Exception as error:
                span.set_status(Status(status_code=StatusCode.ERROR, description=str(error)))
                
                raise
        
            
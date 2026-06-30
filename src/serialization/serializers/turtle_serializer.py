from owlready2 import Thing, Ontology, default_world
from serialization.base_serializer import BaseSerializer
from rdflib import Graph, URIRef

class TurtleSerializer(BaseSerializer):
    """
    Concrete implementation of BaseSerializer that serializes OWL individuals and ontologies into Turtle format.
    """

    def process_ontology(self, ontology: Ontology) -> str:
        """
        Serializes an OWL ontology into a Turtle string.

        Args:
            ontology: The OWL ontology to serialize.

        Returns:
            A Turtle string representation of the ontology.
        """
        graph = default_world.as_rdflib_graph(ontology)
        return graph.serialize(format="turtle")

    def process_individual(self, individual: Thing) -> str:
        """
        Serializes an OWL individual into a Turtle string.

        Args:
            individual: The OWL individual to serialize.

        Returns:
            A Turtle string representation of the individual.
        """
        full_graph = default_world.as_rdflib_graph()
        individual_graph = Graph()
        subject_iri = URIRef(individual.iri)
        
        for s, p, o in full_graph.triples((subject_iri, None, None)):
            individual_graph.add((s, p, o))
            
        return individual_graph.serialize(format="turtle")
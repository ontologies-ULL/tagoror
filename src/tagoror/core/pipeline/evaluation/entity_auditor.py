from abc import ABC, abstractmethod
from typing import Any

from tagoror.core.models import ExecutionSummary

from owlready2 import Thing, Ontology
from opentelemetry import trace 

class EntityAuditor(ABC):
    """
    Abstract base class for evaluation strategies.
    """

    @abstractmethod
    async def run(self, individual: Thing, base_ontology: Ontology, cache_name: str = None) -> ExecutionSummary:
        """
        Evaluates an OWL individual and returns an execution summary.

        Args:
            individual: The OWL individual extracted by OntologyExtractor.
            base_ontology: The base ontology for context.
            cache_name: The name of the context cache to use.
            tracer: The OpenTelemetry tracer for instrumentation.

        Returns:
            ExecutionSummary with results and aggregated real metrics.
        """
        pass
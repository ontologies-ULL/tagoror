from abc import ABC, abstractmethod
from typing import Any

from core.models import ExecutionSummary

from owlready2 import Thing

class EntityAuditor(ABC):
    """
    Abstract base class for evaluation strategies.
    """

    @abstractmethod
    async def run(self, individual: Thing) -> ExecutionSummary:
        """
        Evaluates an OWL individual and returns an execution summary.

        Args:
            individual: The OWL individual extracted by OntologyExtractor.

        Returns:
            ExecutionSummary with results and aggregated real metrics.
        """
        pass
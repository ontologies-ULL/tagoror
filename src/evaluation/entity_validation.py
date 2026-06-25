"""
Define the base class for all validation strategies
"""

from abc import ABC, abstractmethod

from core.models import ExecutionSummary

class ValidationStrategy(ABC):
    """
    Base interface to the strategies evaluators
    """

    @abstractmethod
    async def evaluate(self, owl_entity: str) -> ExecutionSummary:
        """
        Evaluate the entity using a type of strategy and generate a report about it.

        Args:
            owl_entity (str): Entity Owl to evaluate

        Returns:
            ExecutionSummary: All data recolected for the final result 
        """
        pass

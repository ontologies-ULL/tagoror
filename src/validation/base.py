"""
Define the base class for all validation strategies
"""

import asyncio
from abc import ABC, abstractmethod

class ValidationStrategy(ABC):
    """
    Base interface to the strategies evaluators
    """

    @abstractmethod
    async def evaluate(self, owl_entity: str) -> ValidationReport:
        """
        Evaluate the entity using a type of strategy and generate a report about it.

        Args:
            owl_entity (str): Entity Owl to evaluate

        Returns:
            ValidationReport: All data recolected for the final result 
        """
        pass

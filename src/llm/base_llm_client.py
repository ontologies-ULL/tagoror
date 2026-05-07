"""

"""

from . import config
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Base interface to the LLM clients
    """

    @abstractmethod
    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Query the LLM with a given payload and return the response.

        Args:
            payload (LLMPayload): The payload to send to the LLM

        Returns:
            LLMResponse: The response from the LLM
        """
        pass

"""

"""

from .models import LLMPayload, LLMResponse
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Base interface to the LLM clients
    """

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Query the LLM with a given payload and return the response.

        Args:
            payload (LLMPayload): The payload to send to the LLM

        Returns:
            LLMResponse: The response from the LLM
        """
        try:
            response = await self._query(payload)
            return response
        except Exception as error:
            pass

    @abstractmethod
    async def _query(self, payload: LLMPayload) -> LLMResponse:
        """
        Abstract method to be implemented by subclasses to perform the actual query to the LLM.

        Args:
            payload (LLMPayload): The payload to send to the LLM

        Returns:
            LLMResponse: The response from the LLM
        """
        pass

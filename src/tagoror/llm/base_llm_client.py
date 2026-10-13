"""

"""

import random

from abc import ABC, abstractmethod

from .models import LLMPayload, LLMResponse

class BaseLLMClient(ABC):
    """
    Base interface to the LLM clients
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

        self._tracer = trace.get_tracer("tagoror.llm")
        self._meter  = metrics.get_meter("tagoror.llm")
        self._logger = get_logger("tagoror.llm")

        self._request_counter = self._meter.create_counter(
            name="llm.client.requests",
            description="Total number of LLM API requests",
            unit="1",
        )
        self._duration_histogram = self._meter.create_histogram(
            name="llm.client.duration",
            description="Duration of LLM API requests",
            unit="ms",
        )
        self._token_counter = self._meter.create_counter(
            name="llm.client.tokens",
            description="Total tokens consumed by LLM API requests",
            unit="1",
        )
        self._cost_counter = self._meter.create_counter(
            name="llm.client.cost",
            description="Estimated cost of LLM API requests",
            unit="USD",
        )

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

"""

"""

import random
from asyncio import sleep
from .models import LLMPayload, LLMResponse
from .config import RetryPolicyConfig, BackoffStrategy
from exceptions import TransientNetworkException, LLMParseException
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Base interface to the LLM clients
    """

    _retry_config: RetryPolicyConfig | None = None

    def set_retry_config(self, retry_config: RetryPolicyConfig | None) -> None:
        self._retry_config = retry_config

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Query the LLM with a given payload and return the response.

        Args:
            payload (LLMPayload): The payload to send to the LLM

        Returns:
            LLMResponse: The response from the LLM
        """
        retry_config = self._retry_config
        if not retry_config:
            return await self._query(payload)

        for attempt in range(retry_config.max_retries):
            try:
                return await self._query(payload)
            except (TransientNetworkException, LLMParseException):
                if attempt == retry_config.max_retries - 1:
                    raise
                delay = self._calculate_retry_delay(retry_config, attempt)
                await sleep(delay)

        raise RuntimeError("Retry loop terminated unexpectedly")

    def _calculate_retry_delay(self, retry_config: RetryPolicyConfig, attempt: int) -> float:
        base_delay = retry_config.delay_between_retries
        strategy = retry_config.backoff_strategy

        if strategy == BackoffStrategy.FIXED:
            return float(base_delay)
        if strategy == BackoffStrategy.EXPONENTIAL:
            return float(base_delay * (2 ** attempt))
        if strategy == BackoffStrategy.JITTER:
            max_delay = base_delay * (2 ** attempt)
            return random.uniform(base_delay, max_delay)

        return float(base_delay)

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

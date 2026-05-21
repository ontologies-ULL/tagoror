"""
Define the policy for retrying LLM calls in case of failures, including the number of retries and the delay between retries.
"""

import random

from .base_llm_client import BaseLLMClient
from .config import RetryPolicyConfig, BackoffStrategy
from exceptions import TransientNetworkException, LLMParseException
from .models import LLMPayload, LLMResponse

from asyncio import sleep

class RetryPolicy:
    """
    A class that defines the policy for retrying LLM calls in case of failures, 
    including the number of retries and the delay between retries.
    """
    def __init__(self, llm_client: BaseLLMClient, config: RetryPolicyConfig):
        self.llm_client = llm_client
        self.config = config

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Execute an LLM call with the defined retry policy.
        """
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await self.llm_client.query(payload)
            except (TransientNetworkException, LLMParseException) as error:
                if attempt == self.config.max_retries:
                    raise error
                delay = self._calculate_delay(attempt)
                await sleep(delay)

    def _calculate_delay(self, current_attempt: int) -> float:
        """
        Calcula el tiempo de espera (sleep) basado en la estrategia configurada.
        current_attempt: El número de intento que acaba de fallar (1, 2, 3...)
        """
        base_delay = self.config.delay_between_retries
        strategy = self.config.backoff_strategy

        if strategy == BackoffStrategy.FIXED:
            return float(base_delay)
        elif strategy == BackoffStrategy.EXPONENTIAL:
            return float(base_delay * (2 ** current_attempt))
        elif strategy == BackoffStrategy.JITTER:
            max_delay = base_delay * (2 ** current_attempt)
            return random.uniform(base_delay, max_delay)
            
        return float(base_delay)

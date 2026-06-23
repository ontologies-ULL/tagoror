import random
from asyncio import sleep

from .base_llm_client import BaseLLMClient
from .config import RetryPolicyConfig, BackoffStrategy
from .models import LLMPayload, LLMResponse
from exceptions import TransientNetworkException, LLMParseException

class RetryableLLMClient(BaseLLMClient):
    """
    Proxy class that implements the BaseLLMClient interface and adds transparent
    resilience (retries) to any underlying LLM client.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config: RetryPolicyConfig):
        """
        Initializes the retryable client proxy.

        Args:
            llm_client (BaseLLMClient): The concrete LLM client instance to wrap.
            config (RetryPolicyConfig): The configuration defining the retry policy.
        """
        self.llm_client = llm_client
        self.config = config

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Executes an LLM call using the defined retry policy.
        Propagates the specific domain exception if the maximum number of attempts is reached.

        Args:
            payload (LLMPayload): The payload to send to the LLM provider.

        Returns:
            LLMResponse: The successful response from the underlying LLM client.

        Raises:
            TransientNetworkException: If a network error persists after all retries are exhausted.
            LLMParseException: If the response parsing fails after all retries are exhausted.
            RuntimeError: If the retry loop terminates unexpectedly.
        """
        max_attempts = self.config.max_retries
        
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.llm_client.query(payload)
            except (TransientNetworkException, LLMParseException) as error:
                if attempt == max_attempts:
                    raise error
                
                delay = self._calculate_delay(attempt)
                await sleep(delay)
                
        raise RuntimeError("The retry loop terminated unexpectedly.")

    def _calculate_delay(self, current_attempt: int) -> float:
        """
        Calculates the wait time (sleep) based on the configured backoff strategy.

        Args:
            current_attempt (int): The current failed attempt number (e.g., 1, 2, 3).

        Returns:
            float: The calculated delay in seconds before the next retry.
        """
        base_delay = self.config.delay_between_retries
        strategy = self.config.backoff_strategy
        exponential_base = self.config.exponential_base

        if strategy == BackoffStrategy.FIXED:
            return float(base_delay)
        elif strategy == BackoffStrategy.EXPONENTIAL:
            return float(base_delay * (exponential_base ** current_attempt))
        elif strategy == BackoffStrategy.JITTER:
            max_delay = base_delay * (exponential_base ** current_attempt)
            return random.uniform(base_delay, max_delay)
            
        return float(base_delay)
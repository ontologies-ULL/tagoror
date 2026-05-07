"""
Define the policy for retrying LLM calls in case of failures, including the number of retries and the delay between retries.
"""

from .base_llm_client import BaseLLMClient
from .config import RetryPolicyConfig
from .exceptions import TransientNetworkError, LLMParseException
from .models import LLMPayload, LLMResponse

from asyncio import sleep

class RetryPolicy:
    """
    A class that defines the policy for retrying LLM calls in case of failures, including the number of retries and the delay between retries.
    """
    def __init__(self, llm_client: BaseLLMClient, config: RetryPolicyConfig):
        self.llm_client = llm_client
        self.config = config

    async def query(self, payload: LLMPayload) -> LLMResponse:
        """
        Execute an LLM call with the defined retry policy.

        Args:
            payload (LLMPayload): The payload to send to the LLM

        Returns:
            LLMResponse: The response from the LLM
        """
        for attempt in range(self.config.max_retries):
            try:
                response = await self.llm_client.query(payload)
                return response
            except Exception as error:
                if attempt < self.config.max_retries - 1:
                    await sleep(self.config.delay_between_retries)
                else:
                    raise error

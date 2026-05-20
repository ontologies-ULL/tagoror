"""

"""

from dataclasses import dataclass

@dataclass(frozen=True)
class RetryPolicyConfig:
    """
    Represents the retry policy for an LLM request, including the number of retries and the backoff strategy.
    """
    max_retries: int
    delay_between_retries: int  # in seconds
    backoff_strategy: str 

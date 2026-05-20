from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMPayload:
    user_prompt: str
    system_prompt: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    json_mode: bool = True

@dataclass
class LLMResponse:
    raw_content: str
    tokens_consumed: int
    duration_ms: int
    cost: float

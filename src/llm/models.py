from typing import Optional
from pydantic import BaseModel

class LLMPayload(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    json_mode: bool = True
    allow_web_search: bool = False

class LLMResponse(BaseModel):
    raw_content: str
    tokens_consumed: int
    duration_ms: int
    cost: float

from typing import Optional
from pydantic import BaseModel
from tagoror.core.models import TaskStatus
from pydantic import BaseModel, Field, field_validator

class TaskValidationContent(BaseModel):
    status: TaskStatus = Field(
        ..., 
        description="State of the validation task, which can be success, failure, skipped, or partial success."
    )
    findings: list[str] = Field(
        default_factory=list,
        description="List of specific errors found during the validation task."
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step explanation of the validation process, highlighting any potential hallucinations."
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str) -> TaskStatus:
        """Normalize unstructured status inputs into strict TaskStatus enum values."""
        if isinstance(value, str):
            val = value.lower().strip()
            if val in ["success", "compliant", "passed", "pass", "ok", "true"]:
                return TaskStatus.SUCCESS
            if val in ["failure", "failed", "non-compliant", "non_compliant", "error", "false"]:
                return TaskStatus.FAILURE
        return TaskStatus.SKIPPED

class LLMPayload(BaseModel):
    user_prompt: str
    system_prompt: str | None = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    response_model: type[BaseModel] | None = None
    allow_web_search: bool = False
    cache_name: str | None = None

class LLMResponse(BaseModel):
    raw_content: str | None = None
    parsed: TaskValidationContent | None = None
    tokens_consumed: int
    duration_ms: int
    cost: float

"""
Defines data models for validation results, execution metrics, and validation reports. These models are used to structure the data returned by the validation process, including individual task results, overall metrics, and system summaries. The models also include methods for assessing the success of the validation process based on individual task outcomes.
"""

from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Optional

class TaskStatus(Enum):
    """
    Indicates the status of a validation task, which can be success, failure, skipped, or partial success.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    PARTIAL_SUCCESS = "partial_success"

class ExecutionMetrics(BaseModel):
    """
    Represents the execution metrics for a validation task, including duration, cost, and tokens consumed.
    """
    duration_ms: int
    cost: float
    tokens_consumed: int


class TaskOutcome(BaseModel):
    """
    Represents the outcome of a single validation task, including its ID, status, and findings.
    Metrics and rawResponse are optional for error cases.
    """
    task_id: str
    status: TaskStatus
    findings: list[str]
    metrics: Optional[ExecutionMetrics] = None
    rawResponse: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class ExecutionSummary(BaseModel):
    """
    Represents the overall summary of a validation execution, including the individual ID, 
    timestamp, results of all tasks, total metrics, and a system summary.
    """
    individual_id: str
    timestamp: str
    results: list[TaskOutcome]
    total_metrics: ExecutionMetrics
    system_summary: str

    model_config = ConfigDict(frozen=True)

    def is_successful(self) -> bool:
        return all(result.status == TaskStatus.SUCCESS for result in self.results)


class PromptRegistry(BaseModel):
    """
    Represents a registry of prompts used in the validation process, mapping task IDs to their corresponding prompts.
    """
    loaded_prompts: dict[str, str]

    model_config = ConfigDict(frozen=True)

    def get_prompt(self, task_id: str) -> str:
        return self.loaded_prompts.get(task_id, "")
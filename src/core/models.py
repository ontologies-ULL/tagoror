"""
Defines data models for validation results, execution metrics, and validation reports. These models are used to structure the data returned by the validation process, including individual task results, overall metrics, and system summaries. The models also include methods for assessing the success of the validation process based on individual task outcomes.
"""

from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    """
    Indicates the status of a validation task, which can be success, failure, skipped, or partial success.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    PARTIAL_SUCCESS = "partial_success"

@dataclass(frozen=True)
class ExcecutionMetrics:
    """
    Represents the execution metrics for a validation task, including duration, cost, and tokens consumed.
    """
    duration_ms: int
    cost: float
    tokens_consumed: int

@dataclass(frozen=True)
class TaskOutcome:
    """
    Represents the outcome of a single validation task, including its ID, status, findings, execution metrics, and raw response.
    """
    task_id: str
    status: TaskStatus
    findings: list[str]
    metrics: ExcecutionMetrics
    rawResponse: str

@dataclass(frozen=True)
class ExecutionSummary:
    """
    Represents the overall summary of a validation execution, including the individual ID, timestamp, results of all tasks, total metrics, and a system summary. It also includes a method to determine if the entire validation process was successful based on the status of individual tasks.
    """
    individual_id: str
    timestamp: str
    results: list[TaskOutcome]
    total_metrics: ExcecutionMetrics
    system_summary: str

    def is_successful(self) -> bool:
        return all(result.status == TaskStatus.SUCCESS for result in self.results)

@dataclass(frozen=True)
class PromptRegistry:
    """
    Represents a registry of prompts used in the validation process, mapping task IDs to their corresponding prompts.
    """
    loaded_prompts: dict[str, str]

    def get_prompt(self, task_id: str) -> str:
        return self.loaded_prompts.get(task_id, "")

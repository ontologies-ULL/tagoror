from collections import Counter
from core.models import TaskOutcome, TaskStatus

class ConsensusResolver:
    """
    Applies Majority Voting (Self-Consistency) to a list of LLM task outcomes.
    It determines the final status by democratic vote and aggregates the findings.
    """

    def resolve(self, task_id: str, branch_results: list[tuple[TaskOutcome, float]]) -> TaskOutcome:
        if not branch_results:
            return None

        statuses = [outcome.status for outcome, temp in branch_results]
        vote_counts = Counter(statuses).most_common()

        if len(vote_counts) > 1 and vote_counts[0][1] == vote_counts[1][1]:
            return None

        majority_status = vote_counts[0][0]
        majority_findings = []
        for outcome, temp in branch_results:
            if outcome.status == majority_status:
                majority_findings.extend(outcome.findings)

        unique_findings = list(dict.fromkeys(majority_findings))

        return TaskOutcome(
            task_id=task_id,
            status=majority_status,
            findings=unique_findings
        )
from dataclasses import dataclass


@dataclass
class TriageResult:
    score: float
    decision: str


def classify_score(
    score: float,
    review_threshold: float,
    defect_threshold: float,
) -> TriageResult:

    if score < review_threshold:
        decision = "AUTO_NORMAL"

    elif score < defect_threshold:
        decision = "HUMAN_REVIEW"

    else:
        decision = "AUTO_DEFECT"

    return TriageResult(
        score=float(score),
        decision=decision,
    )
from inspection.localization.triage import (
    classify_score,
)


def test_auto_normal():

    result = classify_score(
        score=1.0,
        review_threshold=2.0,
        defect_threshold=3.0,
    )

    assert result.decision == "AUTO_NORMAL"


def test_human_review():

    result = classify_score(
        score=2.5,
        review_threshold=2.0,
        defect_threshold=3.0,
    )

    assert result.decision == "HUMAN_REVIEW"


def test_auto_defect():

    result = classify_score(
        score=4.0,
        review_threshold=2.0,
        defect_threshold=3.0,
    )

    assert result.decision == "AUTO_DEFECT"
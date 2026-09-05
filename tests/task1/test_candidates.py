from fashion.task1.candidates import (
    TASK1_GENTLE_WEIGHTED_CANDIDATE,
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
)


def test_three_cnn_candidates_have_distinct_explicit_identities() -> None:
    candidates = (
        TASK1_NO_AUG_CANDIDATE,
        TASK1_MILD_AUG_CANDIDATE,
        TASK1_GENTLE_WEIGHTED_CANDIDATE,
    )
    assert [candidate.candidate_id for candidate in candidates] == [
        "task1_cnn_no_aug_unweighted_v1",
        "task1_cnn_mild_aug_unweighted_v1",
        "task1_cnn_no_aug_sqrt_weighted_v1",
    ]
    assert TASK1_GENTLE_WEIGHTED_CANDIDATE.preprocessing == TASK1_NO_AUG_CANDIDATE.preprocessing
    assert TASK1_GENTLE_WEIGHTED_CANDIDATE.loss.loss_id != TASK1_NO_AUG_CANDIDATE.loss.loss_id

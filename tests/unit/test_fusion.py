"""Reciprocal Rank Fusion is the core of hybrid retrieval — tested directly."""

from app.rag.pipeline import _rrf_fuse


def test_single_ranking_preserves_order():
    ranking = [("d1", 0), ("d1", 1), ("d2", 0)]
    fused = _rrf_fuse([ranking])
    assert [(d, i) for d, i, _ in fused] == ranking


def test_agreement_between_legs_wins():
    vector = [("d1", 0), ("d2", 0), ("d3", 0)]
    keyword = [("d3", 0), ("d1", 0), ("d9", 0)]
    fused = _rrf_fuse([vector, keyword])
    # d1 ranks 1st and 2nd; nothing else appears high in both.
    assert (fused[0][0], fused[0][1]) == ("d1", 0)


def test_item_in_only_one_leg_still_surfaces():
    fused = _rrf_fuse([[("d1", 0)], [("d2", 0)]])
    keys = {(d, i) for d, i, _ in fused}
    assert keys == {("d1", 0), ("d2", 0)}


def test_scores_are_descending():
    fused = _rrf_fuse([[("d1", 0), ("d2", 0), ("d3", 0)]])
    scores = [s for _, _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_empty_input_is_empty_output():
    assert _rrf_fuse([]) == []
    assert _rrf_fuse([[]]) == []


def test_damping_constant_reduces_top_rank_dominance():
    ranking = [("a", 0), ("b", 0)]
    tight = _rrf_fuse([ranking], k=1)
    loose = _rrf_fuse([ranking], k=1000)
    tight_gap = tight[0][2] - tight[1][2]
    loose_gap = loose[0][2] - loose[1][2]
    assert loose_gap < tight_gap

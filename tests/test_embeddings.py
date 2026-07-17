from app.rag.embeddings import HashEmbedder


def test_embeddings_are_unit_normalised():
    vec = HashEmbedder(dim=64).embed_one("retrieval augmented generation")
    assert abs(sum(v * v for v in vec) ** 0.5 - 1.0) < 1e-6


def test_embedding_is_deterministic():
    e = HashEmbedder(dim=64)
    assert e.embed_one("same text") == e.embed_one("same text")


def test_similar_text_scores_higher_than_unrelated():
    e = HashEmbedder(dim=256)
    q = e.embed_one("kubernetes autoscaling policy")
    near = e.embed_one("kubernetes autoscaling")
    far = e.embed_one("banana bread recipe")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(q, near) > dot(q, far)


def test_batch_matches_single():
    e = HashEmbedder(dim=32)
    batch = e.embed(["one", "two"])
    assert batch[0] == e.embed_one("one")
    assert len(batch) == 2


def test_empty_string_does_not_divide_by_zero():
    assert HashEmbedder(dim=16).embed_one("") == [0.0] * 16

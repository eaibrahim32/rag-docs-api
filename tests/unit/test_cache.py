from app.db.redis_cache import cache_key


def test_key_is_stable_for_same_inputs():
    assert cache_key("What is RAG?", 5, None) == cache_key("What is RAG?", 5, None)


def test_key_normalises_case_and_padding():
    assert cache_key("  What Is RAG?  ", 5, None) == cache_key("what is rag?", 5, None)


def test_key_varies_with_top_k():
    assert cache_key("q", 5, None) != cache_key("q", 8, None)


def test_key_varies_with_document_scope():
    assert cache_key("q", 5, ["a"]) != cache_key("q", 5, ["b"])


def test_document_scope_is_order_independent():
    assert cache_key("q", 5, ["a", "b"]) == cache_key("q", 5, ["b", "a"])


def test_key_is_namespaced():
    assert cache_key("q", 5, None).startswith("answer:")

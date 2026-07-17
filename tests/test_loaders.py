import pytest

from app.core.errors import UnsupportedFileType
from app.rag.loaders import load, normalise


def test_txt_roundtrip():
    text, pages = load("notes.txt", b"hello  world")
    assert text == "hello world"
    assert pages == []


def test_markdown_is_supported():
    text, _ = load("readme.md", b"# Title\n\nBody text.")
    assert "Title" in text


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileType):
        load("archive.zip", b"PK\x03\x04")


def test_extensionless_file_raises():
    with pytest.raises(UnsupportedFileType):
        load("Dockerfile", b"FROM python")


def test_normalise_collapses_whitespace():
    assert normalise("a   b\t\tc") == "a b c"


def test_normalise_rejoins_hyphenated_line_breaks():
    assert "retrieval" in normalise("retrie-\nval augmented")


def test_normalise_caps_blank_lines():
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_utf8_errors_do_not_crash():
    text, _ = load("bad.txt", b"caf\xe9 latte")
    assert "latte" in text

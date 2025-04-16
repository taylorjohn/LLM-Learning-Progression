import pytest
from collections import Counter
from src.text_processing import tokenize, count_words

# --- Tokenize Tests ---

def test_tokenize_simple():
    assert tokenize("hello world") == ["hello", "world"]

def test_tokenize_case():
    assert tokenize("Hello World") == ["hello", "world"]

def test_tokenize_punctuation():
    assert tokenize("Hello, world! Go.") == ["hello", "world", "go"]

def test_tokenize_mixed():
    # Matches the Rust test case
    assert tokenize("alpha beta-- gamma delta! dash-dash") == ["alpha", "beta", "gamma", "delta", "dash-dash"]

def test_tokenize_empty():
    assert tokenize("") == []

def test_tokenize_only_punctuation():
    assert tokenize(", . ! -- ??") == []

def test_tokenize_numbers_and_symbols():
    assert tokenize("word1 123 !!another-word?? 456") == ["word1", "123", "another-word", "456"]

def test_tokenize_internal_apostrophe():
    assert tokenize("it's a test isn't it?") == ["it's", "a", "test", "isn't", "it"]

# --- Count Words Tests ---

def test_count_words_simple():
    assert count_words("hello world hello") == Counter({"hello": 2, "world": 1})

def test_count_words_case_insensitive():
    assert count_words("Hello hello HELLO") == Counter({"hello": 3})

def test_count_words_with_punctuation():
    # Matches the Rust test case
    assert count_words("Go, team, go! Go team?") == Counter({"go": 3, "team": 2})

def test_count_words_empty_after_tokenize():
    assert count_words("! , . --") == Counter()

def test_count_words_mixed():
    text = "Test, test... this is a test--or isn't it?"
    # Corrected expectation based on tokenize rule (keep internal hyphens)
    expected = Counter({"test": 2, "this": 1, "is": 1, "a": 1, "test--or": 1, "isn't": 1, "it": 1})
    assert count_words(text) == expected 
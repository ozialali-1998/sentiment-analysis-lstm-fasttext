"""Utility functions for tweet preprocessing.

This module contains reusable helpers for cleaning tweets, tokenization,
stopword filtering, and preparing a pandas DataFrame before model training.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence, Set

import pandas as pd


# Minimal Indonesian + English stopword list so the project can run offline
# without requiring additional corpora downloads (e.g. NLTK assets).
DEFAULT_STOPWORDS: Set[str] = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "ini",
    "itu",
    "aku",
    "kamu",
    "dia",
    "mereka",
    "kita",
    "saya",
    "the",
    "a",
    "an",
    "is",
    "are",
    "am",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "it",
    "this",
    "that",
}


def clean_tweet_text(text: str) -> str:
    """Clean raw tweet text into normalized lowercase text.

    Main steps:
    1. Lowercase all characters.
    2. Remove URLs.
    3. Remove mentions and hashtags symbols (#topic -> topic).
    4. Remove numbers and punctuation.
    5. Collapse multiple spaces.
    """

    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"@\w+", " ", text)  # remove @mentions
    text = re.sub(r"#", " ", text)  # keep hashtag word, remove # symbol
    text = re.sub(r"\d+", " ", text)  # remove numeric characters

    # Remove punctuation using translation table for performance/readability.
    punctuation_table = str.maketrans("", "", string.punctuation)
    text = text.translate(punctuation_table)

    # Remove any non-letter remnants and normalize extra spaces.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """Split cleaned text into tokens.

    Tokenization is whitespace-based for simplicity and reproducibility.
    """

    if not text:
        return []
    return text.split()


def remove_stopwords(tokens: Sequence[str], stopwords: Set[str] | None = None) -> List[str]:
    """Remove stopwords from token list.

    Args:
        tokens: Tokenized words from a tweet.
        stopwords: Optional custom stopword set. If None, use DEFAULT_STOPWORDS.
    """

    active_stopwords = stopwords or DEFAULT_STOPWORDS
    return [token for token in tokens if token not in active_stopwords]


def preprocess_tweet(text: str, stopwords: Set[str] | None = None) -> List[str]:
    """Run full preprocessing pipeline for one tweet.

    Returns token list after cleaning, tokenization, and stopword removal.
    """

    cleaned = clean_tweet_text(text)
    tokens = tokenize_text(cleaned)
    return remove_stopwords(tokens, stopwords=stopwords)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "tweet",
    stopwords: Set[str] | None = None,
) -> pd.DataFrame:
    """Preprocess tweets in DataFrame and add helper columns.

    Output columns:
    - cleaned_text: cleaned plain text
    - tokens: list of tokens
    - processed_text: tokens joined by space for model/tokenizer input
    """

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    processed_df = df.copy()

    # Apply each preprocessing stage explicitly for easier debugging and
    # for educational clarity in an academic project.
    processed_df["cleaned_text"] = processed_df[text_column].astype(str).apply(clean_tweet_text)
    processed_df["tokens"] = processed_df["cleaned_text"].apply(tokenize_text)
    processed_df["tokens"] = processed_df["tokens"].apply(
        lambda token_list: remove_stopwords(token_list, stopwords=stopwords)
    )
    processed_df["processed_text"] = processed_df["tokens"].apply(lambda token_list: " ".join(token_list))

    return processed_df


def tokens_for_fasttext(token_series: Iterable[Sequence[str]]) -> List[List[str]]:
    """Convert DataFrame token series into list-of-lists for FastText training."""

    return [list(tokens) for tokens in token_series]

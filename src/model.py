"""Modeling utilities for FastText embeddings and LSTM sentiment classifier."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from gensim.models import FastText
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocessing import preprocess_tweet


LABEL_TO_ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


def train_fasttext_model(
    tokenized_sentences: Iterable[Sequence[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 20,
) -> FastText:
    """Train FastText embedding model from tokenized sentences."""

    sentences = [list(tokens) for tokens in tokenized_sentences if tokens]
    if not sentences:
        raise ValueError("No valid tokenized sentences provided for FastText training.")

    fasttext_model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )
    fasttext_model.build_vocab(sentences)
    fasttext_model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return fasttext_model


def build_tokenizer(texts: Sequence[str], max_num_words: int = 20000) -> Tokenizer:
    """Fit Keras tokenizer on processed text corpus."""

    tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def create_embedding_matrix(
    tokenizer: Tokenizer,
    fasttext_model: FastText,
    max_num_words: int,
    embedding_dim: int,
) -> np.ndarray:
    """Build embedding matrix where row i is the FastText vector for token i."""

    vocab_size = min(max_num_words, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in fasttext_model.wv:
            embedding_matrix[idx] = fasttext_model.wv[word]

    return embedding_matrix


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int,
    max_sequence_len: int,
    embedding_matrix: np.ndarray,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    trainable_embedding: bool = False,
):
    """Build and compile LSTM model for 3-class sentiment classification."""

    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_len,
                weights=[embedding_matrix],
                trainable=trainable_embedding,
            ),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def text_to_padded_sequence(tokenizer: Tokenizer, texts: Sequence[str], max_sequence_len: int) -> np.ndarray:
    """Convert raw text list into padded integer sequence matrix."""

    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_sequence_len, padding="post", truncating="post")


@dataclass
class SentimentArtifacts:
    """Container for serialized assets used during inference."""

    model_path: Path
    tokenizer_path: Path
    config_path: Path
    fasttext_path: Path


class SentimentInference:
    """Load trained assets and provide prediction API for one text."""

    def __init__(self, artifacts: SentimentArtifacts):
        self.model = load_model(artifacts.model_path)
        with artifacts.tokenizer_path.open("rb") as f:
            self.tokenizer: Tokenizer = pickle.load(f)
        with artifacts.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.fasttext_model = FastText.load(str(artifacts.fasttext_path))

        self.max_sequence_len = int(self.config["max_sequence_len"])
        self.embedding_dim = int(self.config["embedding_dim"])

    def predict_sentiment(self, text: str) -> Dict[str, object]:
        """Predict sentiment for a single tweet.

        Returns dictionary with:
        - label: predicted class label
        - probabilities: class probabilities for negative/neutral/positive
        """

        tokens = preprocess_tweet(text)
        processed_text = " ".join(tokens)
        padded = text_to_padded_sequence(self.tokenizer, [processed_text], self.max_sequence_len)

        probs = self.model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        return {
            "label": ID_TO_LABEL[pred_idx],
            "probabilities": {
                ID_TO_LABEL[i]: float(prob) for i, prob in enumerate(probs)
            },
            "tokens": tokens,
        }

    def extract_feature_vector(self, text: str) -> Dict[str, float]:
        """Extract numerical features needed for downstream K-Means clustering."""

        prediction = self.predict_sentiment(text)
        probs = prediction["probabilities"]
        tokens: List[str] = prediction["tokens"]

        if tokens:
            token_vectors = [self.fasttext_model.wv[token] for token in tokens]
            avg_embedding = np.mean(token_vectors, axis=0)
        else:
            avg_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        sentiment_strength = probs["positive"] - probs["negative"]

        features = {
            "prob_negative": probs["negative"],
            "prob_neutral": probs["neutral"],
            "prob_positive": probs["positive"],
            "sentiment_strength": float(sentiment_strength),
        }

        for i, value in enumerate(avg_embedding):
            features[f"emb_{i}"] = float(value)

        return features

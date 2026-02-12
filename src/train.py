"""Training pipeline for tweet sentiment analysis using FastText + LSTM.

Example:
    python -m src.train --data_path data/tweets.csv --text_col tweet --label_col sentiment
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.model import (
    LABEL_TO_ID,
    SentimentArtifacts,
    SentimentInference,
    build_lstm_model,
    build_tokenizer,
    create_embedding_matrix,
    text_to_padded_sequence,
    train_fasttext_model,
)
from src.preprocessing import preprocess_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastText + LSTM sentiment model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset file.")
    parser.add_argument("--text_col", type=str, default="tweet", help="Tweet text column name.")
    parser.add_argument("--label_col", type=str, default="sentiment", help="Sentiment label column name.")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Directory to save trained artifacts.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_num_words", type=int, default=20000, help="Tokenizer vocabulary cap.")
    parser.add_argument("--max_sequence_len", type=int, default=50, help="Padded sequence length.")
    parser.add_argument("--embedding_dim", type=int, default=100, help="FastText vector size.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for LSTM.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    return parser.parse_args()


def validate_labels(df: pd.DataFrame, label_col: str) -> None:
    """Validate that labels are exactly negative/neutral/positive."""

    unique_labels = set(df[label_col].astype(str).str.lower().unique())
    allowed = set(LABEL_TO_ID.keys())
    if not unique_labels.issubset(allowed):
        raise ValueError(
            f"Invalid labels found: {unique_labels - allowed}. "
            f"Expected labels only from {sorted(allowed)}."
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset and basic quality checks.
    df = pd.read_csv(args.data_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"Dataset must contain '{args.text_col}' and '{args.label_col}' columns. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[args.text_col, args.label_col]].dropna().reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(str).str.lower()
    validate_labels(df, args.label_col)

    # 2) Preprocess tweets into cleaned tokens and processed text.
    processed_df = preprocess_dataframe(df, text_column=args.text_col)

    # 3) Train FastText embeddings on tokenized tweets.
    fasttext_model = train_fasttext_model(
        tokenized_sentences=processed_df["tokens"],
        vector_size=args.embedding_dim,
        epochs=20,
    )

    # 4) Encode labels and split train/test.
    y = processed_df[args.label_col].map(LABEL_TO_ID).values
    X_text = processed_df["processed_text"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 5) Tokenizer + sequence preparation for LSTM input.
    tokenizer = build_tokenizer(X_train_text, max_num_words=args.max_num_words)
    X_train = text_to_padded_sequence(tokenizer, X_train_text, args.max_sequence_len)
    X_test = text_to_padded_sequence(tokenizer, X_test_text, args.max_sequence_len)

    # 6) Build embedding matrix based on FastText vectors.
    embedding_matrix = create_embedding_matrix(
        tokenizer=tokenizer,
        fasttext_model=fasttext_model,
        max_num_words=args.max_num_words,
        embedding_dim=args.embedding_dim,
    )

    vocab_size = embedding_matrix.shape[0]

    # 7) Build, train, and evaluate LSTM classifier.
    model = build_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        max_sequence_len=args.max_sequence_len,
        embedding_matrix=embedding_matrix,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n=== Evaluation Summary ===")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["negative", "neutral", "positive"],
            digits=4,
        )
    )

    # 8) Save all artifacts for future inference and clustering.
    model_path = output_dir / "sentiment_lstm.keras"
    tokenizer_path = output_dir / "tokenizer.pkl"
    fasttext_path = output_dir / "fasttext.model"
    config_path = output_dir / "config.json"

    model.save(model_path)
    with tokenizer_path.open("wb") as f:
        pickle.dump(tokenizer, f)
    fasttext_model.save(str(fasttext_path))

    config = {
        "max_sequence_len": args.max_sequence_len,
        "embedding_dim": args.embedding_dim,
        "max_num_words": args.max_num_words,
        "label_to_id": LABEL_TO_ID,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 9) Extract numerical features for every tweet and save feature table.
    inference = SentimentInference(
        SentimentArtifacts(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            fasttext_path=fasttext_path,
        )
    )

    feature_rows = []
    for original_text in processed_df[args.text_col].astype(str):
        features = inference.extract_feature_vector(original_text)
        prediction = inference.predict_sentiment(original_text)

        row = {
            "tweet": original_text,
            "predicted_label": prediction["label"],
            **features,
        }
        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    features_path = output_dir / "tweet_features.csv"
    features_df.to_csv(features_path, index=False)

    print(f"\nArtifacts saved to: {output_dir.resolve()}")
    print(f"Feature table saved to: {features_path.resolve()}")


if __name__ == "__main__":
    main()

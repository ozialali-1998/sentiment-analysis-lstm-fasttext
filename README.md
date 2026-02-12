# Sentiment Analysis Tweet (FastText + LSTM + K-Means)

Proyek ini berisi pipeline end-to-end untuk skripsi analisis sentimen tweet:
1. Preprocessing tweet (cleaning, tokenizing, stopword removal)
2. Training FastText embedding
3. Klasifikasi sentimen dengan model LSTM (negative, neutral, positive)
4. Ekstraksi fitur numerik (probabilitas kelas, sentiment strength, embedding)
5. Clustering K-Means + visualisasi 2D

## Struktur Folder

```text
.
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── clustering.py
├── requirements.txt
└── README.md
```

## Format Dataset

Siapkan file CSV (contoh: `data/tweets.csv`) minimal dengan kolom:
- `tweet`: teks tweet
- `sentiment`: label (`negative`, `neutral`, `positive`)

Contoh isi:

```csv
tweet,sentiment
"Pelayanan cepat dan ramah sekali!",positive
"Biasa saja, tidak terlalu bagus.",neutral
"Kecewa, aplikasinya sering error.",negative
```

## Instalasi (Windows Python 3.10+)

1. Buat virtual environment:

```bash
python -m venv .venv
```

2. Aktifkan virtual environment (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

3. Install dependency:

```bash
pip install -r requirements.txt
```

## Training Model

Jalankan perintah berikut dari root project:

```bash
python -m src.train --data_path data/tweets.csv --text_col tweet --label_col sentiment --output_dir artifacts --epochs 8 --batch_size 32
```

Output utama di folder `artifacts/`:
- `sentiment_lstm.keras` (model LSTM)
- `tokenizer.pkl` (tokenizer Keras)
- `fasttext.model` (embedding FastText)
- `config.json` (konfigurasi training)
- `tweet_features.csv` (fitur numerik per tweet)

## Clustering K-Means

Setelah training selesai, jalankan:

```bash
python -m src.clustering --features_path artifacts/tweet_features.csv --n_clusters 3 --output_dir artifacts
```

Output clustering:
- `tweet_features_clustered.csv`
- `kmeans_scatter.png`

## Contoh Inferensi `predict_sentiment(text)`

Berikut contoh pemakaian API inferensi untuk 1 tweet:

```python
from pathlib import Path
from src.model import SentimentArtifacts, SentimentInference

artifacts = SentimentArtifacts(
    model_path=Path("artifacts/sentiment_lstm.keras"),
    tokenizer_path=Path("artifacts/tokenizer.pkl"),
    config_path=Path("artifacts/config.json"),
    fasttext_path=Path("artifacts/fasttext.model"),
)

inference = SentimentInference(artifacts)
result = inference.predict_sentiment("Aplikasinya sangat membantu dan fiturnya bagus")
print(result)
# contoh output:
# {
#   'label': 'positive',
#   'probabilities': {'negative': 0.02, 'neutral': 0.12, 'positive': 0.86},
#   'tokens': ['aplikasinya', 'sangat', 'membantu', 'fiturnya', 'bagus']
# }
```

## Ringkasan Tiap File

- `src/preprocessing.py`: fungsi pembersihan tweet, tokenisasi, stopword removal, preprocessing DataFrame.
- `src/model.py`: training FastText, pembuatan embedding matrix, arsitektur LSTM, kelas inferensi + ekstraksi fitur.
- `src/train.py`: script training end-to-end dari CSV hingga simpan artifact + fitur numerik.
- `src/clustering.py`: script clustering K-Means pada fitur numerik + visualisasi scatter 2D (PCA).


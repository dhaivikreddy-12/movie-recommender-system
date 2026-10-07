# 🎬 Movie Recommender System

> *"Netflix, but mine — a recommender that learns what people like from nothing but their ratings."*

The project where matrix factorization finally clicked for me. Trained on **100,836 real MovieLens ratings** from 610 users, it learns a latent feature space for both users and movies — no genre metadata, no plot summaries, just "who rated what".

## What this project does

- Downloads the real MovieLens ml-latest-small dataset (100,836 ratings, 610 users, 9,724 movies).
- Encodes user and movie IDs into a dense index space.
- Trains a matrix factorization model with learned user/movie embeddings and bias terms.
- Uses a **70/10/20 train/val/test split with early stopping** — the split that made this model actually work.
- Compares against an honest train-mean baseline instead of a flattering one.
- Recommends top-N unseen movies for any real MovieLens userId.

## The dataset

[MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) — 100,836 ratings, 0.5–5 stars, 610 users, 9,724 movies.

| File | Columns |
|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` |
| `movies.csv` | `movieId`, `title`, `genres` |

## How to run it

```bash
pip install -r requirements.txt

python -m src.train                          # train, evaluate, recommend for userId 1
python -m src.train --recommend 4 --top 10   # recommendations for another user
```

## Project structure

```
movie-recommender-system/
├── src/
│   ├── data_gen.py   # MovieLens download + cache
│   ├── model.py      # MatrixFactorization
│   └── train.py      # training, early stopping, top-N
├── data/
│   ├── ratings.csv
│   └── movies.csv
├── tests/
├── requirements.txt
└── README.md
```

## What I learned

- That the *global bias* initialisation matters enormously. Starting it at 0 instead of the mean rating cost me an entire round of debugging.
- Why early stopping on a validation split was the difference between a model that beat baseline and one that didn't.
- That comparing against the **train mean** is the honest baseline. Using the test mean flatters the model and made me briefly think I'd built something useless.
- What collaborative filtering can and can't see: it knows patterns, not reasons.

## Results

70/10/20 split, early stopping at epoch 10 (best val RMSE 0.8966):

| Metric | Value |
|---|---|
| Validation RMSE | 0.8966 |
| **Test RMSE** | **0.9041** |
| Test MAE | 0.6981 |
| Train-mean baseline RMSE | 1.0529 |
| **Improvement over baseline** | **14.1%** |

Beating a constant-prediction baseline by 14% on a 0.5–5 rating scale is a real result for an unregularised matrix factorization, and the top-N recommendations are recognisably sensible.

---

*Built with Python, PyTorch, pandas. Real rating data, honestly measured.*

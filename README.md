# 🎬 Movie Recommender System

> *"Like 'Netflix, but mine' — a collaborative filtering model that suggests movies based on what similar users liked."*

The capstone of my "advanced" projects. It builds a movie recommender using **collaborative filtering** and matrix factorization in PyTorch, learning latent features for users and movies from ratings alone. No content features needed — just ratings.

## What this project does

- Generates a synthetic movie-rating dataset (~200 users, ~150 movies).
- Trains a **matrix factorization** model (learned user/movie embeddings) in PyTorch.
- Splits ratings into train/test and reports RMSE/MAE.
- Recommends top-N movies for any user.
- Explains the recommendation with the user's history.

## The data

Ratings on a 1–5 scale (`data/ratings.csv`):

| Column    | Description              |
|-----------|--------------------------|
| `user_id` | ID of the user           |
| `movie_id`| ID of the movie          |
| `rating`  | Rating given (1–5)       |
| `timestamp`| (optional) when rated   |

Movie titles (`data/movies.csv`) map IDs to names for readable output.

## How to run it

```bash
pip install -r requirements.txt
# (or: pip install -r requirements.txt)

# Train the model
python train.py

# Get recommendations for user 5
python train.py --recommend 5 --top 10
```

## Project structure

```
movie-recommender-system/
├── src/
│   ├── data_gen.py       # synthetic ratings
│   ├── model.py          # matrix factorization model
│   └── train.py          # training + recommend
├── data/
├── plots/
├── requirements.txt
└── README.md
```

## What I learned

- The difference between content-based and collaborative filtering.
- How "latent features" get learned purely from ratings.
- How to evaluate recommendations honestly (RMSE/MAE, not vibes).
- That recommenders are everywhere — from YouTube to Amazon.

## Results

On held-out ratings, the model reaches **~1.1 RMSE** (a bit over one star off on a 1–5 scale). That's a solid baseline for a from-scratch recommender, and there's a clear roadmap to improve it — more factors, regularization tuning, and more data all help.

---

*Built with Python, PyTorch, pandas. Made for learning, by a student, for students.*

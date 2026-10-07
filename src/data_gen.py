"""Generate a synthetic movie ratings dataset."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(2025)

n_users = 200
n_movies = 150

rng_u = rng.normal(0, 0.5, (n_users, 10))
rng_m = rng.normal(0, 0.5, (n_movies, 10))
user_bias = rng.normal(0, 0.4, n_users)
movie_bias = rng.normal(0, 0.4, n_movies)

ratings = []
for user_id in range(n_users):
    n_rated = rng.integers(20, 80)
    movie_ids = rng.choice(n_movies, n_rated, replace=False)
    for movie_id in movie_ids:
        pred = 3.5 + user_bias[user_id] + movie_bias[movie_id] \
               + float(rng_u[user_id] @ rng_m[movie_id])
        rating = pred + rng.normal(0, 0.35)
        rating = int(np.clip(round(rating), 1, 5))
        ts = rng.integers(1600000000, 1700000000)
        ratings.append((user_id, movie_id, rating, ts))

df = pd.DataFrame(ratings, columns=["user_id", "movie_id", "rating", "timestamp"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/ratings.csv", index=False)

titles = [f"Movie {i+1}" for i in range(n_movies)]
movies_df = pd.DataFrame({"movie_id": range(n_movies), "title": titles})
movies_df.to_csv("data/movies.csv", index=False)

print(f"Generated {len(df)} ratings across {n_users} users and {n_movies} movies.")
print(df.head())
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

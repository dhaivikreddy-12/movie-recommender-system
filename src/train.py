"""Train the recommender on real MovieLens ratings and print top-N suggestions."""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MatrixFactorization, rmse
from src.data_gen import load

torch.manual_seed(0)

ratings, movies = load()
print(f"Ratings: {len(ratings)} | Users: {ratings.userId.nunique()} | Movies: {ratings.movieId.nunique()}")

# Keep only users with a reasonable history so the factorised space is learnable.
min_ratings_per_user = 20
counts = ratings.userId.value_counts()
keep_users = counts[counts >= min_ratings_per_user].index
ratings = ratings[ratings.userId.isin(keep_users)]
print(f"After filtering users with >= {min_ratings_per_user} ratings: {len(ratings)} rows, {ratings.userId.nunique()} users")

user_ids, uidx = np.unique(ratings.userId, return_inverse=True)
movie_ids, midx = np.unique(ratings.movieId, return_inverse=True)
n_users, n_movies = len(user_ids), len(movie_ids)
print(f"Encoded space: {n_users} users x {n_movies} movies")

u = torch.tensor(uidx, dtype=torch.long)
m = torch.tensor(midx, dtype=torch.long)
t = torch.tensor(ratings["rating"].to_numpy(), dtype=torch.float32)

perm = torch.randperm(len(t))
# 70 / 10 / 20 train/val/test so early stopping has something to watch.
n_train, n_val = int(0.70 * len(t)), int(0.10 * len(t))
train = TensorDataset(u[perm[:n_train]], m[perm[:n_train]], t[perm[:n_train]])
val = TensorDataset(u[perm[n_train:n_train + n_val]], m[perm[n_train:n_train + n_val]], t[perm[n_train:n_train + n_val]])
test = TensorDataset(u[perm[n_train + n_val:]], m[perm[n_train + n_val:]], t[perm[n_train + n_val:]])

train_loader = DataLoader(train, batch_size=8192, shuffle=True)
val_loader = DataLoader(val, batch_size=8192)
test_loader = DataLoader(test, batch_size=8192)

model = MatrixFactorization(n_users, n_movies, n_factors=32, init_mean=float(t[perm[:n_train]].mean()))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

EPOCHS = 40
PATIENCE = 5
best_val = float("inf")
wait = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for ub, mb, tb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(ub, mb), tb)
        loss.backward()
        optimizer.step()
        total += loss.item() * ub.size(0)
    train_rmse = np.sqrt(total / len(train))

    model.eval()
    with torch.no_grad():
        vp = torch.cat([model(ub, mb) for ub, mb, _ in val_loader])
        va = torch.cat([tb for _, _, tb in val_loader])
    val_rmse = rmse(vp, va)

    if val_rmse < best_val - 1e-4:
        best_val, wait = val_rmse, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>2}/{EPOCHS} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}")

    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best val RMSE {best_val:.4f})")
        break

model.load_state_dict(best_state)
print(f"\nBest validation RMSE: {best_val:.4f}")

model.eval()
with torch.no_grad():
    preds = torch.cat([model(ub, mb) for ub, mb, _ in test_loader])
    actual = torch.cat([tb for _, _, tb in test_loader])
test_rmse = rmse(preds, actual)
test_mae = float(torch.mean(torch.abs(preds - actual)))
# Honest baseline: predict the training mean for everything.
train_mean = float(t[perm[:n_train]].mean())
baseline_rmse = float(torch.sqrt(torch.mean((actual - train_mean) ** 2)))
print(
    f"\nTest RMSE {test_rmse:.4f} | Test MAE {test_mae:.4f} "
    f"| Train-mean baseline RMSE {baseline_rmse:.4f}"
)
print(f"Improvement over baseline: {(1 - test_rmse / baseline_rmse) * 100:.1f}%")

# Map encoded indices back to real MovieLens ids.
encoded_to_user = {i: int(user_ids[i]) for i in range(n_users)}
title_by_movie = dict(zip(movies["movieId"], movies["title"]))
seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()


def recommend(user_id, top_n=10, trained_model=None):
    """Top-N unseen movies for a real MovieLens userId."""
    model = trained_model if trained_model is not None else globals()["model"]
    model.eval()
    if user_id not in encoded_to_user.values():
        raise ValueError(f"userId {user_id} is not in the filtered dataset")
    idx = list(encoded_to_user.values()).index(user_id)
    real_movie_ids = movie_ids.tolist()
    already = seen.get(user_id, set())

    with torch.no_grad():
        users = torch.full((n_movies,), idx, dtype=torch.long)
        movies_idx = torch.arange(n_movies, dtype=torch.long)
        scores = model(users, movies_idx)

    order = torch.argsort(scores, descending=True).tolist()
    out = []
    for i in order:
        real_id = real_movie_ids[i]
        if real_id in already:
            continue
        out.append((real_id, title_by_movie.get(real_id, f"Movie {real_id}")))
        if len(out) >= top_n:
            break
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommender (MovieLens)")
    parser.add_argument("--recommend", type=int, default=1, help="MovieLens userId")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    print(f"\nTop {args.top} recommendations for userId {args.recommend}:")
    for i, (mid, title) in enumerate(recommend(args.recommend, args.top), 1):
        print(f"  {i:>2}. {title}  (id {mid})")

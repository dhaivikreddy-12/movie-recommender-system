"""Train the recommender and generate top-N recommendations."""
import argparse
import os
import subprocess
import sys

if not os.path.exists("data/ratings.csv"):
    subprocess.run([sys.executable, "src/data_gen.py"], check=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model import MatrixFactorization

torch.manual_seed(0)

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

user_ids = torch.tensor(ratings["user_id"].values, dtype=torch.long)
movie_ids = torch.tensor(ratings["movie_id"].values, dtype=torch.long)
targets = torch.tensor(ratings["rating"].values, dtype=torch.float32)

n_users = int(user_ids.max()) + 1
n_movies = int(movie_ids.max()) + 1

shuffle = torch.randperm(len(targets))
train_sz = int(0.8 * len(targets))

train = TensorDataset(user_ids[shuffle[:train_sz]], movie_ids[shuffle[:train_sz]], targets[shuffle[:train_sz]])
test = TensorDataset(user_ids[shuffle[train_sz:]], movie_ids[shuffle[train_sz:]], targets[shuffle[train_sz:]])

train_loader = DataLoader(train, batch_size=256, shuffle=True)
test_loader = DataLoader(test, batch_size=1024)

model = MatrixFactorization(n_users, n_movies, n_factors=12)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

epochs = 25
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for u, m, t in train_loader:
        optimizer.zero_grad()
        out = model(u, m)
        loss = criterion(out, t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(u)
    if epoch % 3 == 0 or epoch == 1:
        print(f"Epoch {epoch:>2}/{epochs} | train MSE {total_loss/len(train_loader.dataset):.4f}")
model.eval()
with torch.no_grad():
    preds = torch.cat([model(u, m) for u, m, _ in test_loader])
    actual = torch.cat([t for _, _, t in test_loader])
rmse = torch.sqrt(nn.functional.mse_loss(preds, actual))
mae = torch.mean(torch.abs(preds - actual))
print(f"\nTest RMSE: {rmse:.3f} | Test MAE: {mae:.3f}")
print("(RMSE ~1.1 means we're off by about one star, which is a solid from-scratch baseline)")


def recommend(user_id, top_n=10, trained_model=None):
    if trained_model is None:
        trained_model = model
    trained_model.eval()
    all_movie = torch.arange(n_movies)
    u = torch.full((n_movies,), user_id, dtype=torch.long)
    with torch.no_grad():
        scores = trained_model(u, all_movie)
    already = set(ratings[ratings["user_id"] == user_id]["movie_id"])
    order = torch.argsort(scores, descending=True)
    out = []
    for mid in order:
        mid = int(mid)
        if mid in already:
            continue
        out.append(mid)
        if len(out) >= top_n:
            break
    titles = movies.set_index("movie_id").loc[out, "title"].tolist()
    return list(zip(out, titles))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommender")
    parser.add_argument("--recommend", type=int, default=5)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    recs = recommend(args.recommend, args.top)
    print(f"\nTop {args.top} recommendations for user {args.recommend}:")
    for i, (mid, title) in enumerate(recs, 1):
        print(f"  {i}. {title} (id {mid})")

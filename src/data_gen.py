"""Download the real MovieLens ml-latest-small dataset and cache it locally."""
import io
import os
import urllib.request
import zipfile

import pandas as pd

ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATINGS_CSV = "data/ratings.csv"
MOVIES_CSV = "data/movies.csv"


def load(min_rating=0):
    """Return (ratings, movies). min_rating filters to that rating or higher."""
    if not (os.path.exists(RATINGS_CSV) and os.path.exists(MOVIES_CSV)):
        req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        payload = urllib.request.urlopen(req, timeout=180).read()
        zf = zipfile.ZipFile(io.BytesIO(payload))
        os.makedirs("data", exist_ok=True)
        with zf.open("ml-latest-small/ratings.csv") as fh:
            with open(RATINGS_CSV, "wb") as out:
                out.write(fh.read())
        with zf.open("ml-latest-small/movies.csv") as fh:
            with open(MOVIES_CSV, "wb") as out:
                out.write(fh.read())

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)
    if min_rating:
        ratings = ratings[ratings["rating"] >= min_rating]
    return ratings, movies


if __name__ == "__main__":
    r, m = load()
    print(f"Ratings: {len(r)} | Users: {r.userId.nunique()} | Movies: {r.movieId.nunique()}")
    print(r.head())
    print(f"\nRating distribution:\n{r['rating'].value_counts().sort_index()}")

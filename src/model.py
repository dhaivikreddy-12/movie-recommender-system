"""Matrix factorization recommender trained on real MovieLens ratings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    """Learns a low-rank latent space for users and movies, plus biases."""

    def __init__(self, n_users, n_movies, n_factors=32, init_mean=3.9):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        # Start the global offset at the observed mean rating so the
        # optimisation does not have to learn the baseline from zero.
        self.global_bias = nn.Parameter(torch.tensor([float(init_mean)]))
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.movie_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids):
        dot = (self.user_factors(user_ids) * self.movie_factors(movie_ids)).sum(dim=1)
        return dot + self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze() + self.global_bias


def rmse(pred, actual):
    return float(torch.sqrt(F.mse_loss(pred, actual)))

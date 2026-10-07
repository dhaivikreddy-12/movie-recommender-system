"""Matrix factorization model for collaborative filtering."""
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=12, bias=True):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.bias = bias
        if bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.movie_bias = nn.Embedding(n_movies, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids):
        u = self.user_factors(user_ids)
        m = self.movie_factors(movie_ids)
        pred = (u * m).sum(dim=1, keepdim=True)
        if self.bias:
            pred = pred + self.user_bias(user_ids) + self.movie_bias(movie_ids) + self.global_bias
        return pred.squeeze(-1)

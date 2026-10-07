import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RATINGS_CSV = ROOT / "data" / "ratings.csv"
MOVIES_CSV = ROOT / "data" / "movies.csv"


def _require_cache():
    if not (RATINGS_CSV.exists() and MOVIES_CSV.exists()):
        pytest.skip("MovieLens cache missing; run the loader once to cache it")


def _model(**kwargs):
    from src.model import MatrixFactorization

    params = {"n_users": 50, "n_movies": 200, "n_factors": 8, "init_mean": 3.9}
    params.update(kwargs)
    return MatrixFactorization(**params)


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file(), "README.md is missing"
    assert (ROOT / "LICENSE").is_file(), "LICENSE is missing"


def test_data_gen_module_points_at_local_cache():
    from src import data_gen

    assert callable(data_gen.load)
    assert data_gen.RATINGS_CSV == "data/ratings.csv"
    assert data_gen.MOVIES_CSV == "data/movies.csv"
    assert data_gen.ZIP_URL.startswith("https://")


def test_cached_movielens_tables_load_without_network(monkeypatch):
    _require_cache()
    from src.data_gen import load

    monkeypatch.chdir(ROOT)
    ratings, movies = load()

    assert len(ratings) == 100836, f"expected ml-latest-small ratings, got {len(ratings)}"
    assert list(ratings.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert list(movies.columns) == ["movieId", "title", "genres"]
    assert len(movies) == 9742
    assert ratings.userId.nunique() == 610
    assert ratings.rating.min() >= 0.5 and ratings.rating.max() <= 5.0
    assert ratings.rating.mean() == pytest.approx(3.5, abs=0.05)


def test_min_rating_filter_drops_low_scores(monkeypatch):
    _require_cache()
    from src.data_gen import load

    monkeypatch.chdir(ROOT)
    all_ratings, _ = load()
    high, _ = load(min_rating=4)

    assert len(high) < len(all_ratings)
    assert high.rating.min() >= 4.0
    assert len(high) + int((all_ratings.rating < 4).sum()) == len(all_ratings)


def test_global_bias_starts_at_init_mean():
    model = _model()
    assert model.global_bias.shape == (1,)
    assert model.global_bias.item() == pytest.approx(3.9, abs=1e-5)

    other = _model(init_mean=1.5)
    assert other.global_bias.item() == pytest.approx(1.5, abs=1e-5)


def test_default_init_mean_is_the_dataset_mean_rating():
    from src.model import MatrixFactorization

    assert MatrixFactorization(10, 10).global_bias.item() == pytest.approx(3.9, abs=1e-5)


def test_embedding_biases_start_at_zero():
    import torch

    model = _model()
    assert torch.equal(model.user_bias.weight, torch.zeros_like(model.user_bias.weight))
    assert torch.equal(model.movie_bias.weight, torch.zeros_like(model.movie_bias.weight))
    assert model.user_factors.weight.shape == (50, 8)
    assert model.movie_factors.weight.shape == (200, 8)


def test_forward_pass_returns_finite_vector():
    import torch

    model = _model()
    users = torch.randint(0, 50, (64,))
    movies = torch.randint(0, 200, (64,))
    out = model(users, movies)

    assert out.shape == (64,)
    assert out.dtype.is_floating_point
    assert torch.isfinite(out).all()
    assert out.abs().max() < 100.0
    assert torch.allclose(out, model(users, movies))


def test_backward_pass_populates_finite_gradients():
    import torch

    model = _model()
    users = torch.randint(0, 50, (64,))
    movies = torch.randint(0, 200, (64,))
    model(users, movies).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"

    assert model.user_factors.weight.grad.shape == (50, 8)
    assert model.movie_factors.weight.grad.shape == (200, 8)
    assert torch.isfinite(model.user_factors.weight.grad).all()
    assert torch.isfinite(model.movie_factors.weight.grad).all()
    assert model.global_bias.grad.item() == pytest.approx(64.0, rel=1e-5)


def test_rmse_is_zero_for_identical_tensors():
    import torch

    from src.model import rmse

    assert rmse(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 0.0
    assert rmse(torch.tensor([5.0]), torch.tensor([5.0])) == 0.0


def test_rmse_matches_hand_computed_error():
    import math

    import torch

    from src.model import rmse

    assert rmse(torch.tensor([2.0]), torch.tensor([4.0])) == pytest.approx(2.0, abs=1e-6)
    assert rmse(torch.tensor([2.0]), torch.tensor([4.0])) == 2.0

    value = rmse(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0]))
    assert value == pytest.approx(math.sqrt(14 / 3), rel=1e-5)
    assert isinstance(value, float)


def test_rmse_returns_a_detached_python_float():
    import warnings

    import torch

    from src.model import MatrixFactorization, rmse

    model = MatrixFactorization(n_users=10, n_movies=10, n_factors=4)
    users = torch.randint(0, 10, (20,))
    movies = torch.randint(0, 10, (20,))
    predictions = model(users, movies)
    assert predictions.requires_grad

    targets = torch.full((20,), 3.0)
    expected = float(torch.sqrt(((predictions.detach() - targets) ** 2).mean()))

    # rmse() wraps the result in float(), so it detaches from the graph.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        value = rmse(predictions, targets)

    assert type(value) is float
    assert value == pytest.approx(expected, rel=1e-6)
    # Predictions start near init_mean=3.9, so the error against 3.0 is ~0.9.
    assert 0.85 < value < 1.0
    assert model.global_bias.grad is None, "float() return should detach from autograd"

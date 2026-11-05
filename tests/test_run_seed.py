"""APGI-REP-002: one recorded seed reproduces every stochastic path."""

import APGI_System as A


def _noise_draws():
    return [A.run_rng().normal() for _ in range(5)] + [A.run_rng().random()]


def test_same_seed_same_draws_and_seed_is_recorded():
    assert A.set_run_seed(123) == 123 and A.get_run_seed() == 123
    a = _noise_draws()
    A.set_run_seed(123)
    assert _noise_draws() == a
    A.set_run_seed(124)
    assert _noise_draws() != a


def test_unseeded_run_still_records_a_replayable_seed():
    A._RUN_SEED, A._RUN_RNG = None, None
    seed = A.get_run_seed()
    first = _noise_draws()
    A.set_run_seed(seed)
    assert _noise_draws() == first


def test_invalid_seed_rejected():
    import pytest

    for bad in (-1, 1.5, True, "7"):
        with pytest.raises(ValueError):
            A.set_run_seed(bad)


def test_no_unseeded_global_numpy_random_left_in_module():
    import re
    from pathlib import Path

    src = Path(A.__file__).read_text(encoding="utf-8")
    assert not re.search(r"np\.random\.(normal|random|rand|randn|choice|uniform)\(", src)
    assert "default_rng()" not in src.replace("np.random.default_rng(seed)", "")

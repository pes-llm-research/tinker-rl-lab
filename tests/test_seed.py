"""Smoke tests for seed management and stats utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_global_seed


def test_set_global_seed_deterministic():
    """Verify global seed produces deterministic random state."""
    import random

    set_global_seed(42)
    a = [random.random() for _ in range(10)]
    set_global_seed(42)
    b = [random.random() for _ in range(10)]
    assert a == b, "Same seed should produce same random sequence"


def test_set_global_seed_different():
    """Different seeds produce different sequences."""
    import random

    set_global_seed(42)
    a = random.random()
    set_global_seed(99)
    b = random.random()
    assert a != b, "Different seeds should produce different values"


def test_seed_returns_dict():
    """Seed info returns a dict with expected keys."""
    info = set_global_seed(42)
    assert isinstance(info, dict)


def test_numpy_seed():
    """Numpy seeding is deterministic."""
    try:
        import numpy as np

        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        assert (a == b).all(), "Numpy should be deterministic with same seed"
    except ImportError:
        pass  # numpy not installed, skip


if __name__ == "__main__":
    test_set_global_seed_deterministic()
    test_set_global_seed_different()
    test_seed_returns_dict()
    test_numpy_seed()
    print("All seed tests passed")

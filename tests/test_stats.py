"""Smoke tests for statistical analysis utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_stats_import():
    """Stats module imports without error."""
    from utils import stats

    assert hasattr(stats, "__file__")


def test_stats_has_key_functions():
    """Stats module has expected analysis functions."""
    from utils import stats

    source = open(stats.__file__).read()
    expected = ["bootstrap", "confidence", "latex"]
    found = [kw for kw in expected if kw in source.lower()]
    assert len(found) >= 1, f"Stats module should have analysis functions, found: {found}"


if __name__ == "__main__":
    test_stats_import()
    test_stats_has_key_functions()
    print("All stats tests passed")

"""Smoke tests verifying experiment files are well-formed."""

import sys
import os
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "implementations")


def test_experiments_parse():
    """All experiment Python files parse without syntax errors."""
    if not os.path.exists(EXP_DIR):
        return
    errors = []
    for fn in sorted(os.listdir(EXP_DIR)):
        if fn.endswith(".py"):
            path = os.path.join(EXP_DIR, fn)
            try:
                with open(path) as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{fn}: {e}")
    assert not errors, f"Syntax errors in experiments: {errors}"


def test_experiments_have_seed():
    """All TRL experiment files import seed management."""
    if not os.path.exists(EXP_DIR):
        return
    missing = []
    for fn in sorted(os.listdir(EXP_DIR)):
        if fn.endswith(".py") and "trl" in fn:
            with open(os.path.join(EXP_DIR, fn)) as f:
                code = f.read()
            if "seed" not in code.lower():
                missing.append(fn)
    assert not missing, f"Experiments missing seed management: {missing}"


if __name__ == "__main__":
    test_experiments_parse()
    test_experiments_have_seed()
    print("All experiment tests passed")

"""Smoke tests — verify the package imports and the environment is sane.

These are intentionally trivial. Real tests land alongside each module.
"""

import importlib


def test_package_imports():
    """The top-level package must be importable."""
    import neuro_paging

    assert neuro_paging is not None


def test_core_deps_available():
    """Core retrieval and pipeline deps must be importable."""
    for dep in [
        "hnswlib",
        "numpy",
        "langgraph",
        "streamlit",
        "mlxtend",
        "hdbscan",
        "pydantic",
        "loguru",
    ]:
        mod = importlib.import_module(dep)
        assert mod is not None, f"{dep} failed to import"


def test_python_version():
    """We require Python 3.11+."""
    import sys

    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_apache_license_present():
    """LICENSE file exists and contains 'Apache License'."""
    from pathlib import Path

    license_path = Path(__file__).parent.parent / "LICENSE"
    assert license_path.exists(), "LICENSE file missing"
    content = license_path.read_text()
    assert "Apache License" in content, "LICENSE is not Apache-2.0"

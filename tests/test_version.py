"""The version is written in two places; this makes them agree.

`pyproject.toml` is what PyPI and the release tag check read; `__version__` is what
users and the HTTP API report. They drifted once (0.1.0 vs 0.2.0), which is invisible
at runtime and only shows up as a wrong version on an already-published artifact.
"""

import re
import tomllib
from pathlib import Path

import nodewatch

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_dunder_version_matches_pyproject():
    with open(PYPROJECT, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert nodewatch.__version__ == declared, (
        f"__init__.py says {nodewatch.__version__}, pyproject.toml says {declared}"
    )


def test_version_is_pep440_ish():
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]?(a|b|rc|dev)\d+)?", nodewatch.__version__)


def test_changelog_documents_the_current_version():
    """A release whose version has no changelog entry is a release with no notes."""
    changelog = (PYPROJECT.parent / "CHANGELOG.md").read_text()
    assert f"## [{nodewatch.__version__}]" in changelog

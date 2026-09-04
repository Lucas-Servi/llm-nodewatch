"""Tests for CLI _safe_write helper: path safety and overwrite protection."""

import pytest
import typer

from nodewatch.cli import _safe_write


def test_refuses_system_paths(tmp_path):
    """Should refuse to write to system directories."""
    with pytest.raises(typer.Exit):
        _safe_write("/etc/passwd", "malicious")

    with pytest.raises(typer.Exit):
        _safe_write("/usr/lib/hack.so", "malicious")

    with pytest.raises(typer.Exit):
        _safe_write("/proc/self/mem", "malicious")


def test_refuses_overwrite_without_force(tmp_path):
    """Should refuse to overwrite existing file without --force."""
    target = tmp_path / "existing.txt"
    target.write_text("original")

    with pytest.raises(typer.Exit):
        _safe_write(str(target), "new content", force=False)

    # Original content preserved
    assert target.read_text() == "original"


def test_allows_overwrite_with_force(tmp_path):
    """Should overwrite existing file when force=True."""
    target = tmp_path / "existing.txt"
    target.write_text("original")

    _safe_write(str(target), "new content", force=True)
    assert target.read_text() == "new content"


def test_writes_new_file(tmp_path):
    """Should write to a new file path without issues."""
    target = tmp_path / "subdir" / "output.json"
    _safe_write(str(target), '{"data": true}')
    assert target.read_text() == '{"data": true}'


def test_expands_tilde(tmp_path, monkeypatch):
    """Should expand ~ in paths."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _safe_write("~/test_output.txt", "hello")
    assert (tmp_path / "test_output.txt").read_text() == "hello"

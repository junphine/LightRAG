"""Shared fixtures for the Gremlin offline test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_gremlin_workspace_env(monkeypatch):
    """Keep GREMLIN_WORKSPACE out of every test unless a test sets it itself.

    ``GremlinStorage.__init__`` gives the environment variable priority over
    the constructor argument, so a stray value on the developer's machine
    would silently rename the workspace ("test" -> whatever) and every
    assertion would fail for the wrong reason.
    """
    monkeypatch.delenv("GREMLIN_WORKSPACE", raising=False)
"""Offline test helpers for :mod:`lightrag.kg.gremlin_impl`.

``GremlinStorage`` is a synchronous
:class:`~gremlin_python.driver.driver_remote_connection.DriverRemoteConnection`
wrapper, so the storage methods cannot run against a live server here. What
*can* be asserted offline is every traversal the storage builds: each
``build_*`` function returns a plain ``gremlinpython`` traversal whose
``bytecode.step_instructions`` are deterministic given the workspace/entity
arguments.

Two classifier utilities are provided:

* :func:`signature` -- a *data-insensitive* shape fingerprint of a traversal's
  bytecode (string constants are folded to ``'S'``, predicates to
  ``('P', operator)``, nested traversals stay structural). Used by
  :func:`classify` to map an arbitrary ``_run`` call site back to the
  operation it belongs to.
* :func:`normalize_steps` -- a *verbatim* normalization (only :class:`P`
  instances are unfolded to ``('P', operator, value)`` so they survive
  ``==``, which ``P`` overloads to build predicates). Used to pin the exact
  bytecode of every builder in ``test_gremlin_bytecode.py``.

:class:`_ScriptedRun` is a drop-in replacement for ``storage._run`` that
answers from an ordered ``(kind, response)`` script. Every ``_run`` call site
binds its traversal with a default argument (``lambda t=t: t.toList()``), so
the fake recovers it from ``thunk.__defaults__[0]`` without executing
anything.
"""

from __future__ import annotations

from typing import Any, Callable

from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.traversal import Bytecode, P

from lightrag.kg.gremlin_impl import (
    GremlinStorage,
    build_add_edge,
    build_add_node_props,
    build_add_nodes_batch,
    build_all_edges,
    build_all_labels,
    build_all_nodes,
    build_delete_node,
    build_drop_all,
    build_get_edge,
    build_get_edges_batch,
    build_get_node,
    build_get_node_edges,
    build_get_nodes_batch,
    build_has_edge,
    build_has_node,
    build_has_nodes,
    build_kg_degrees,
    build_kg_edges,
    build_kg_nodes,
    build_node_degree,
    build_node_degrees,
    build_remove_edge,
    build_remove_nodes,
)

WS = "ws1"


def graph_source():
    """An offline ``GraphTraversalSource`` (no remote connection)."""
    return traversal().with_(None)


# --------------------------------------------------------------------------
# Data-insensitive signature (used by classify)
# --------------------------------------------------------------------------


def _sig_val(value: Any):
    if isinstance(value, Bytecode):
        # Nested traversal: recurse into its step_instructions.
        return tuple(_sig_step(step) for step in value.step_instructions)
    if isinstance(value, P):
        return ("P", value.operator)
    if isinstance(value, str):
        return "S"
    if isinstance(value, (int, float, bool)):
        return "N"
    if isinstance(value, (list, tuple)):
        # Plain data list: normalize contents away so the signature is
        # independent of how many inject()/within() entries a call site used.
        return "LIST"
    if isinstance(value, dict):
        return ("D", len(value))
    return type(value).__name__


def _sig_step(step: list[Any]) -> tuple:
    name, *args = step
    return (str(name), tuple(_sig_val(arg) for arg in args))


def signature(traversal_obj) -> tuple:
    """Data-insensitive fingerprint of ``bytecode.step_instructions``."""
    return tuple(_sig_step(step) for step in traversal_obj.bytecode.step_instructions)


# --------------------------------------------------------------------------
# classify: map a built traversal back to its operation name
# --------------------------------------------------------------------------


def _type_to_str(value: Any) -> str:
    return type(value).__name__


def _classify_spec(builder: Callable, args: tuple) -> tuple:
    return signature(builder(graph_source(), WS, *args))


def _build_kind_map() -> dict[tuple, str]:
    """Fixed-shape operations, one canonical traversal each."""
    specs = [
        ("has_node", build_has_node, ("n1",)),
        ("has_nodes", build_has_nodes, (["n1", "n2"],)),
        ("node_degree", build_node_degree, ("n1",)),
        ("node_degrees", build_node_degrees, (["n1", "n2"],)),
        ("has_edge", build_has_edge, ("n1", "n2")),
        ("get_node", build_get_node, ("n1",)),
        ("get_nodes_batch", build_get_nodes_batch, (["n1", "n2"],)),
        ("get_node_edges", build_get_node_edges, ("n1",)),
        ("get_edge", build_get_edge, ("n1", "n2")),
        ("get_edges_batch", build_get_edges_batch, ([("n1", "n2"), ("n3", "n4")],)),
        ("delete_node", build_delete_node, ("n1",)),
        ("remove_nodes", build_remove_nodes, (["n1", "n2"],)),
        ("remove_edge", build_remove_edge, ("n1", "n2")),
        ("drop_all", build_drop_all, ()),
        ("all_labels", build_all_labels, ()),
        ("all_nodes", build_all_nodes, ()),
        ("all_edges", build_all_edges, ()),
        ("kg_degrees", build_kg_degrees, ()),
        ("kg_nodes", build_kg_nodes, (["n1", "n2"],)),
        ("kg_edges", build_kg_edges, (["n1", "n2"],)),
    ]
    mapping: dict[tuple, str] = {}
    for kind, builder, args in specs:
        sig = _classify_spec(builder, args)
        if sig in mapping:
            raise AssertionError(
                f"classify signature collision: {kind!r} and {mapping[sig]!r} "
                f"both produce {sig!r}"
            )
        mapping[sig] = kind
    return mapping


_KIND_MAP = _build_kind_map()

# ``upsert_node`` / ``upsert_edge`` have a variable-length ``property`` tail
# (one step per non-identity key), so they cannot sit in the fixed map. The
# canonical call in ``_classify_spec`` below uses a single dummy key; the
# prefix through the coalesce step is the invariant head every call shares.
_ADD_NODE_PREFIX = _classify_spec(build_add_node_props, ("n", {"dummy": "x"}))[:-1]
_ADD_EDGE_PREFIX = _classify_spec(build_add_edge, ("n1", "n2", {"dummy": 1.0}))[:-1]
_ADD_PREFIXES = {
    "upsert_node": _ADD_NODE_PREFIX,
    "upsert_edge": _ADD_EDGE_PREFIX,
}


def classify(traversal_obj) -> str:
    """Return the operation name for a traversal built by a ``build_*`` function."""
    sig = signature(traversal_obj)
    kind = _KIND_MAP.get(sig)
    if kind is not None:
        return kind

    for kind, prefix in _ADD_PREFIXES.items():
        if sig[: len(prefix)] == prefix and all(
            step_name == "property" for step_name, _ in sig[len(prefix) :]
        ):
            return kind

    raise AssertionError(
        f"classify: unrecognized traversal signature {sig!r}"
    )


# --------------------------------------------------------------------------
# Verbatim normalization (used by the bytecode pinning tests)
# --------------------------------------------------------------------------


def _norm_val(value: Any):
    if isinstance(value, Bytecode):
        # Nested traversal: recurse into its step_instructions.
        return tuple(_norm_step(step) for step in value.step_instructions)
    if isinstance(value, P):
        return ("P", value.operator, _norm_val(value.value))
    if isinstance(value, list):
        return tuple(_norm_val(item) for item in value)
    return value


def _norm_step(step: list[Any]) -> tuple:
    name, *args = step
    return (str(name), tuple(_norm_val(arg) for arg in args))


def normalize_steps(traversal_obj) -> tuple:
    """Verbatim bytecode, with :class:`P` unfolded to comparable tuples."""
    return tuple(_norm_step(step) for step in traversal_obj.bytecode.step_instructions)


# --------------------------------------------------------------------------
# Fake _run
# --------------------------------------------------------------------------


class _ScriptedRun:
    """Drop-in for ``GremlinStorage._run`` driven by an ordered script.

    ``script`` is a list of ``(kind, response)`` pairs. ``kind`` is the
    :func:`classify` name expected at that call site (or ``None`` to accept
    anything). ``response`` is returned verbatim; if it is an ``Exception``
    instance it is raised (matching ``_run``'s contract of propagating
    non-transient errors).
    """

    def __init__(self, script: list[tuple[Any, Any]] | None = None):
        self._script = list(script or [])
        self.calls: list[str] = []

    async def __call__(self, thunk: Callable[[], Any], *args, **kwargs):
        default_args = getattr(thunk, "__defaults__", None)
        if not default_args:
            raise AssertionError(
                "_ScriptedRun: thunk has no default-arg traversal to classify; "
                f"got {thunk!r}"
            )
        t = default_args[0]
        kind = classify(t)
        self.calls.append(kind)
        if not self._script:
            raise AssertionError(
                f"_ScriptedRun: unscripted {kind!r} call (script exhausted)"
            )
        expected, response = self._script.pop(0)
        if expected is not None and expected != kind:
            raise AssertionError(
                f"_ScriptedRun: expected {expected!r} but got {kind!r}"
            )
        if isinstance(response, Exception):
            raise response
        return response


class _ScriptedIterBatches:
    """Drop-in for ``GremlinStorage._iterate_batches`` yielding scripted batches."""

    def __init__(self, batches: list[list[Any]]):
        self._batches = list(batches)
        self.batch_sizes: list[int] = []

    def __call__(self, traversal_factory, batch_size: int):
        self.batch_sizes.append(batch_size)
        return self._agen()

    async def _agen(self):
        for batch in self._batches:
            yield batch


def make_storage(
    script: list[tuple[Any, Any]] | None = None,
    workspace: str = "test",
    namespace: str = "chunk_entity_relation",
):
    """Build a :class:`GremlinStorage` wired for offline testing.

    With ``script`` given, ``_run`` is replaced by a :class:`_ScriptedRun` and
    ``_g`` is an offline traversal source. Without it the storage is left in
    its post-``__init__`` state (``_g``/``_executor`` are ``None``), so the
    real ``_run`` raises the "not initialized" :class:`RuntimeError`.
    """
    storage = GremlinStorage(
        namespace=namespace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace=workspace,
    )
    if script is not None:
        storage._g = graph_source()
        storage._run = _ScriptedRun(script)
    return storage


def run_of(storage):
    """Return the :class:`_ScriptedRun` installed on ``storage`` (assert-checked)."""
    run = getattr(storage, "_run", None)
    assert isinstance(run, _ScriptedRun), "test bug: storage has no _ScriptedRun"
    return run


__all__ = [
    "WS",
    "GremlinStorage",
    "graph_source",
    "signature",
    "classify",
    "normalize_steps",
    "_ScriptedRun",
    "_ScriptedIterBatches",
    "make_storage",
    "run_of",
]
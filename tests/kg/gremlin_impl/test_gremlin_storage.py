"""Offline behavior tests for :class:`GremlinStorage`.

Every storage method is exercised against a :class:`_ScriptedRun` fake that
replaces ``_run``. The fake recovers the traversal from the thunk's default
argument, classifies it, and answers from an ordered ``(kind, response)``
script -- so these tests pin *what the storage does with the response* (the
row-normalization, defaults, early returns, and call sequences) without a
live server.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightrag.kg.gremlin_impl import GremlinStorage
from lightrag.types import KnowledgeGraphEdge
from tests.kg.gremlin_impl._utils import (
    _ScriptedIterBatches,
    make_storage,
    run_of,
)


pytestmark = pytest.mark.offline


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    # The module-level ``lightrag.utils.logger`` sets ``propagate=False``
    # (lightrag/utils.py), so caplog's root handler captures nothing by
    # default. Flip it on for the duration of a test that asserts on warnings.
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


# --------------------------------------------------------------------------
# Existence / degree
# --------------------------------------------------------------------------


async def test_has_node_present_and_absent():
    storage = make_storage([("has_node", [1])])
    assert await storage.has_node("n1") is True
    assert run_of(storage).calls == ["has_node"]

    storage = make_storage([("has_node", [0])])
    assert await storage.has_node("n1") is False

    storage = make_storage([("has_node", [])])
    assert await storage.has_node("n1") is False


async def test_has_nodes_batch():
    storage = make_storage([("has_nodes", ["n1", "n3"])])
    assert await storage.has_nodes_batch(["n1", "n2", "n3"]) == {"n1", "n3"}
    assert run_of(storage).calls == ["has_nodes"]


async def test_has_nodes_batch_empty_short_circuits():
    storage = make_storage([])
    assert await storage.has_nodes_batch([]) == set()
    assert run_of(storage).calls == []


async def test_node_degree():
    storage = make_storage([("node_degree", [3])])
    assert await storage.node_degree("n1") == 3

    storage = make_storage([("node_degree", [])])
    assert await storage.node_degree("n1") == 0


async def test_node_degrees_batch():
    storage = make_storage([("node_degrees", [{"n1": 3, "n3": 0}])])
    assert await storage.node_degrees_batch(["n1", "n2", "n3"]) == {
        "n1": 3,
        "n2": 0,
        "n3": 0,
    }
    assert run_of(storage).calls == ["node_degrees"]


async def test_node_degrees_batch_empty_short_circuits():
    storage = make_storage([])
    assert await storage.node_degrees_batch([]) == {}
    assert run_of(storage).calls == []


async def test_has_edge_present_and_absent():
    storage = make_storage([("has_edge", [1])])
    assert await storage.has_edge("n1", "n2") is True
    assert run_of(storage).calls == ["has_edge"]

    storage = make_storage([("has_edge", [0])])
    assert await storage.has_edge("n1", "n2") is False


async def test_edge_degree_sums_both_node_degrees():
    storage = make_storage([("node_degree", [3]), ("node_degree", [4])])
    assert await storage.edge_degree("n1", "n2") == 7
    assert run_of(storage).calls == ["node_degree", "node_degree"]


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------


async def test_get_node_returns_flattened_props():
    storage = make_storage([("get_node", [{"entity_id": "n1", "description": "d1"}])])
    assert await storage.get_node("n1") == {"entity_id": "n1", "description": "d1"}
    assert run_of(storage).calls == ["get_node"]


async def test_get_node_missing_returns_none():
    storage = make_storage([("get_node", [])])
    assert await storage.get_node("n1") is None

    storage = make_storage([("get_node", [None])])
    assert await storage.get_node("n1") is None


async def test_get_nodes_batch_groups_by_entity_id():
    storage = make_storage(
        [
            (
                "get_nodes_batch",
                [{"n1": {"entity_id": "n1", "description": "d1"}, "n3": {"entity_id": "n3"}}],
            )
        ]
    )
    result = await storage.get_nodes_batch(["n1", "n2", "n3"])
    assert result == {
        "n1": {"entity_id": "n1", "description": "d1"},
        "n3": {"entity_id": "n3"},
    }
    assert run_of(storage).calls == ["get_nodes_batch"]


async def test_get_nodes_batch_empty_short_circuits():
    storage = make_storage([])
    assert await storage.get_nodes_batch([]) == {}
    assert run_of(storage).calls == []


async def test_get_edge_complete():
    storage = make_storage(
        [
            (
                "get_edge",
                [{"weight": 0.5, "source_id": "s", "description": "d", "keywords": "k"}],
            )
        ]
    )
    assert await storage.get_edge("n1", "n2") == {
        "weight": 0.5,
        "source_id": "s",
        "description": "d",
        "keywords": "k",
    }


async def test_get_edge_fills_missing_defaults(caplog, _propagate_lightrag_logger):
    storage = make_storage([("get_edge", [{"weight": 2.5}])])
    result = await storage.get_edge("n1", "n2")
    assert result == {
        "weight": 2.5,
        "source_id": None,
        "description": None,
        "keywords": None,
    }
    # every missing property logs its own warning; weight is present so it does not
    warnings = [str(r.message) for r in caplog.records if "missing property" in str(r.message)]
    assert len(warnings) == 3
    assert {key for w in warnings for key in ("source_id", "description", "keywords") if key in w} == {
        "source_id",
        "description",
        "keywords",
    }


async def test_get_edge_missing_returns_none():
    storage = make_storage([("get_edge", [])])
    assert await storage.get_edge("n1", "n2") is None


async def test_get_edges_batch_keys_by_src_tgt():
    storage = make_storage(
        [
            (
                "get_edges_batch",
                [
                    {"src": "n1", "tgt": "n2", "props": {"weight": 0.5}},
                    {"src": "n3", "tgt": "n4", "props": {}},
                ],
            )
        ]
    )
    result = await storage.get_edges_batch(
        [{"src": "n1", "tgt": "n2"}, {"src": "n3", "tgt": "n4"}]
    )
    assert result == {
        ("n1", "n2"): {"weight": 0.5},
        ("n3", "n4"): {},
    }
    assert run_of(storage).calls == ["get_edges_batch"]


async def test_get_edges_batch_skips_blank_rows():
    storage = make_storage(
        [
            (
                "get_edges_batch",
                [{"src": "n1", "tgt": "n2", "props": {"weight": 0.5}}, {"src": "", "tgt": ""}],
            )
        ]
    )
    result = await storage.get_edges_batch([{"src": "n1", "tgt": "n2"}])
    assert result == {("n1", "n2"): {"weight": 0.5}}


async def test_get_edges_batch_empty_short_circuits():
    storage = make_storage([])
    assert await storage.get_edges_batch([]) == {}
    assert run_of(storage).calls == []


async def test_get_node_edges_returns_pair_list():
    storage = make_storage(
        [
            ("has_node", [1]),
            ("get_node_edges", [{"s": "n1", "t": "n2"}, {"s": "n1", "t": "n3"}]),
        ]
    )
    assert await storage.get_node_edges("n1") == [("n1", "n2"), ("n1", "n3")]
    assert run_of(storage).calls == ["has_node", "get_node_edges"]


async def test_get_node_edges_absent_node_is_none():
    storage = make_storage([("has_node", [0])])
    assert await storage.get_node_edges("n1") is None
    # absent node short-circuits before the second round trip
    assert run_of(storage).calls == ["has_node"]


async def test_get_node_edges_present_but_edgeless_is_empty_list():
    storage = make_storage([("has_node", [1]), ("get_node_edges", [])])
    assert await storage.get_node_edges("n1") == []


async def test_get_node_edges_backend_error_raises():
    storage = make_storage([("has_node", RuntimeError("boom"))])
    with pytest.raises(RuntimeError, match="boom"):
        await storage.get_node_edges("n1")


async def test_get_nodes_edges_batch_maps_none_to_empty():
    storage = make_storage(
        [
            ("has_node", [1]),
            ("get_node_edges", [{"s": "a", "t": "b"}]),
            ("has_node", [0]),
        ]
    )
    result = await storage.get_nodes_edges_batch(["a", "b"])
    assert result == {"a": [("a", "b")], "b": []}


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------


async def test_upsert_node():
    storage = make_storage([("upsert_node", None)])
    await storage.upsert_node("n1", {"description": "d1"})
    assert run_of(storage).calls == ["upsert_node"]


async def test_upsert_nodes_batch():
    storage = make_storage([("upsert_node", None)])
    await storage.upsert_nodes_batch([("n1", {"description": "d1"})])
    assert run_of(storage).calls == ["upsert_node"]


async def test_upsert_nodes_batch_empty_short_circuits():
    storage = make_storage([])
    await storage.upsert_nodes_batch([])
    assert run_of(storage).calls == []


async def test_upsert_edge():
    storage = make_storage([("upsert_edge", None)])
    await storage.upsert_edge("n1", "n2", {"weight": 0.5})
    assert run_of(storage).calls == ["upsert_edge"]


async def test_upsert_edges_batch():
    storage = make_storage([("upsert_edge", None)])
    await storage.upsert_edges_batch([("n1", "n2", {"weight": 0.5})])
    assert run_of(storage).calls == ["upsert_edge"]


async def test_upsert_edges_batch_empty_short_circuits():
    storage = make_storage([])
    await storage.upsert_edges_batch([])
    assert run_of(storage).calls == []


async def test_delete_node():
    storage = make_storage([("delete_node", None)])
    await storage.delete_node("n1")
    assert run_of(storage).calls == ["delete_node"]


async def test_remove_nodes():
    storage = make_storage([("remove_nodes", None)])
    await storage.remove_nodes(["n1", "n2"])
    assert run_of(storage).calls == ["remove_nodes"]


async def test_remove_nodes_empty_short_circuits():
    storage = make_storage([])
    await storage.remove_nodes([])
    assert run_of(storage).calls == []


async def test_remove_edges_loops_per_pair():
    storage = make_storage([("remove_edge", None), ("remove_edge", None)])
    await storage.remove_edges([("n1", "n2"), ("n2", "n3")])
    assert run_of(storage).calls == ["remove_edge", "remove_edge"]


async def test_drop_success():
    storage = make_storage([("drop_all", None)])
    result = await storage.drop()
    assert result["status"] == "success"
    assert "workspace data dropped" in result["message"]
    assert run_of(storage).calls == ["drop_all"]


async def test_drop_error_is_reported_not_raised():
    storage = make_storage([("drop_all", RuntimeError("boom"))])
    result = await storage.drop()
    assert result["status"] == "error"
    assert "boom" in result["message"]
    assert run_of(storage).calls == ["drop_all"]


# --------------------------------------------------------------------------
# Whole-graph iteration
# --------------------------------------------------------------------------


async def test_get_all_labels():
    storage = make_storage([("all_labels", ["n1", "n2", "n3"])])
    assert await storage.get_all_labels() == ["n1", "n2", "n3"]
    assert run_of(storage).calls == ["all_labels"]


async def test_iter_labels_yields_batches():
    storage = make_storage()
    storage._iterate_batches = _ScriptedIterBatches([["n1", "n2"], ["n3"]])
    batches = [batch async for batch in storage.iter_labels(2)]
    assert batches == [["n1", "n2"], ["n3"]]
    # __call__ runs once per iter_labels() invocation, forwarding batch_size
    assert storage._iterate_batches.batch_sizes == [2]


async def test_iter_labels_passes_batch_size_through():
    storage = make_storage()
    storage._iterate_batches = _ScriptedIterBatches([["n1"]])
    [batch async for batch in storage.iter_labels(7)]
    assert storage._iterate_batches.batch_sizes == [7]


async def test_get_all_nodes_injects_id():
    storage = make_storage(
        [
            (
                "all_nodes",
                [{"entity_id": "n1", "description": "d1"}, {"entity_id": "n2"}],
            )
        ]
    )
    result = await storage.get_all_nodes()
    assert result == [
        {"entity_id": "n1", "description": "d1", "id": "n1"},
        {"entity_id": "n2", "id": "n2"},
    ]
    assert run_of(storage).calls == ["all_nodes"]


async def test_get_all_nodes_skips_blank_rows():
    storage = make_storage([("all_nodes", [None])])
    assert await storage.get_all_nodes() == []


async def test_get_all_edges_decorates_src_tgt():
    storage = make_storage(
        [
            (
                "all_edges",
                [{"src": "n1", "tgt": "n2", "props": {"weight": 0.5}}],
            )
        ]
    )
    result = await storage.get_all_edges()
    assert result == [{"weight": 0.5, "source": "n1", "target": "n2"}]
    assert run_of(storage).calls == ["all_edges"]


async def test_iter_edges_yields_decorated_batches():
    storage = make_storage()
    storage._iterate_batches = _ScriptedIterBatches(
        [[{"src": "n1", "tgt": "n2", "props": {"weight": 0.5}}]]
    )
    batches = [batch async for batch in storage.iter_edges(4)]
    assert batches == [[{"weight": 0.5, "source": "n1", "target": "n2"}]]
    assert storage._iterate_batches.batch_sizes == [4]


# --------------------------------------------------------------------------
# Ranking / search
# --------------------------------------------------------------------------


async def test_get_popular_labels_ranks_by_degree_then_label():
    storage = make_storage([("kg_degrees", [{"b": 1, "a": 3, "c": 2, "d": 1}])])
    assert await storage.get_popular_labels(limit=3) == ["a", "c", "b"]
    assert run_of(storage).calls == ["kg_degrees"]


async def test_get_popular_labels_limit_truncates():
    storage = make_storage([("kg_degrees", [{"b": 1, "a": 3, "c": 2}])])
    assert await storage.get_popular_labels(limit=1) == ["a"]


async def test_search_labels_exact_beats_prefix():
    storage = make_storage([("all_labels", ["apple", "applesauce"])])
    assert await storage.search_labels("apple") == ["apple", "applesauce"]


async def test_search_labels_exact_match_scores_highest():
    storage = make_storage([("all_labels", ["app", "application"])])
    # exact match (1000) outranks prefix match (500)
    assert await storage.search_labels("app") == ["app", "application"]


async def test_search_labels_is_case_insensitive():
    storage = make_storage([("all_labels", ["Apple", "banana"])])
    assert await storage.search_labels("APPLE") == ["Apple"]


async def test_search_labels_substring_by_remaining_length():
    storage = make_storage([("all_labels", ["alpha", "alphabet"])])
    # "pha" is a substring of both; shorter label scores higher
    assert await storage.search_labels("pha") == ["alpha", "alphabet"]


async def test_search_labels_blank_query_is_empty_without_call():
    storage = make_storage([])
    assert await storage.search_labels("   ") == []
    assert run_of(storage).calls == []


async def test_search_labels_limit_truncates():
    storage = make_storage([("all_labels", ["a1", "a2", "a3"])])
    assert await storage.search_labels("a", limit=2) == ["a1", "a2"]


# --------------------------------------------------------------------------
# Knowledge graph
# --------------------------------------------------------------------------

# get_knowledge_graph("*") walks the whole store: kg_degrees (rank), then
# kg_nodes + kg_edges for the kept set.
NODE_A = {"entity_id": "a", "description": "da"}
NODE_B = {"entity_id": "b", "description": "db"}
NODE_C = {"entity_id": "c"}
EDGE_A_C = {"src": "a", "tgt": "c", "props": {"weight": 0.5}}


async def test_get_knowledge_graph_star_ranks_fetches_and_builds():
    storage = make_storage(
        [
            ("kg_degrees", [{"a": 2, "c": 1, "b": 1}]),
            ("kg_nodes", [NODE_A, NODE_B, NODE_C]),
            ("kg_edges", [EDGE_A_C]),
        ]
    )
    kg = await storage.get_knowledge_graph("*")
    assert [n.id for n in kg.nodes] == ["a", "b", "c"]
    assert [n.labels for n in kg.nodes] == [["a"], ["b"], ["c"]]
    assert kg.nodes[0].properties == NODE_A
    assert kg.edges == [
        KnowledgeGraphEdge(
            id="a->c",
            type="DIRECTED",
            source="a",
            target="c",
            properties={"weight": 0.5},
        )
    ]
    assert kg.is_truncated is False
    assert run_of(storage).calls == ["kg_degrees", "kg_nodes", "kg_edges"]


async def test_get_knowledge_graph_star_truncates_kept_set():
    # 4 vertices, max 2: degree ranking keeps a, c and flags truncation.
    storage = make_storage(
        [
            ("kg_degrees", [{"a": 2, "c": 2, "b": 1, "d": 0}]),
            ("kg_nodes", [NODE_A, NODE_C]),
            ("kg_edges", [EDGE_A_C]),
        ]
    )
    kg = await storage.get_knowledge_graph("*", max_nodes=2)
    assert [n.id for n in kg.nodes] == ["a", "c"]
    assert kg.is_truncated is True


async def test_get_knowledge_graph_star_empty_degrees_is_empty():
    storage = make_storage([("kg_degrees", [])])
    kg = await storage.get_knowledge_graph("*")
    assert kg.nodes == []
    assert kg.edges == []
    assert kg.is_truncated is False
    assert run_of(storage).calls == ["kg_degrees"]


async def test_get_knowledge_graph_subgraph_bfs_depth_one():
    # A -> B, A -> C with max_depth=1 expands the frontier one hop, then
    # fetches props and the internal edges in the kept set.
    storage = make_storage(
        [
            ("has_node", [1]),
            # BFS get_node_edges("A")
            ("has_node", [1]),
            ("get_node_edges", [{"s": "A", "t": "B"}, {"s": "A", "t": "C"}]),
            # props for kept [A, B, C]
            (
                "get_nodes_batch",
                [
                    {
                        "A": {"entity_id": "A"},
                        "B": {"entity_id": "B"},
                        "C": {"entity_id": "C"},
                    }
                ],
            ),
            # edge collection: A
            ("has_node", [1]),
            ("get_node_edges", [{"s": "A", "t": "B"}, {"s": "A", "t": "C"}]),
            # edge collection: B (edge A->B already seen)
            ("has_node", [1]),
            ("get_node_edges", [{"s": "A", "t": "B"}]),
            # edge collection: C (edge A->C already seen)
            ("has_node", [1]),
            ("get_node_edges", [{"s": "A", "t": "C"}]),
            # edge payloads for the unique pairs
            (
                "get_edges_batch",
                [
                    {"src": "A", "tgt": "B", "props": {"weight": 0.5}},
                    {"src": "A", "tgt": "C", "props": {"weight": 0.5}},
                ],
            ),
        ]
    )
    kg = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=3)
    assert [n.id for n in kg.nodes] == ["A", "B", "C"]
    assert kg.nodes[0].properties == {"entity_id": "A"}
    assert [e.id for e in kg.edges] == ["A->B", "A->C"]
    assert kg.edges[0].source == "A"
    assert kg.edges[0].target == "B"
    assert kg.edges[0].properties == {"weight": 0.5}
    assert kg.is_truncated is False
    assert run_of(storage).calls == [
        "has_node",
        "has_node",
        "get_node_edges",
        "get_nodes_batch",
        "has_node",
        "get_node_edges",
        "has_node",
        "get_node_edges",
        "has_node",
        "get_node_edges",
        "get_edges_batch",
    ]


async def test_get_knowledge_graph_subgraph_missing_start_is_empty():
    # A backward lock-free read: absent start entity never touches the store.
    storage = make_storage([("has_node", [0])])
    kg = await storage.get_knowledge_graph("NoSuchEntity")
    assert kg.nodes == []
    assert kg.edges == []
    assert kg.is_truncated is False
    assert run_of(storage).calls == ["has_node"]


# --------------------------------------------------------------------------
# Runtime wiring
# --------------------------------------------------------------------------


async def test_uninitialized_storage_raises_runtime_error():
    storage = make_storage()
    with pytest.raises(RuntimeError, match="not initialized"):
        await storage._run(lambda: None)


async def test_iter_labels_rejects_non_positive_batch_size():
    storage = make_storage()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        async for _ in storage.iter_labels(0):
            pass


async def test_run_retries_transient_errors_then_succeeds(monkeypatch):
    storage = make_storage()
    storage._g = object()
    storage._executor = ThreadPoolExecutor(max_workers=1)
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("transient")
        return 42

    try:
        assert await storage._run(flaky) == 42
    finally:
        storage._executor.shutdown(wait=False, cancel_futures=True)
    assert sleeps == [0.5, 1.0]
    assert attempts["n"] == 3


async def test_run_gives_up_after_three_attempts(monkeypatch):
    storage = make_storage()
    storage._g = object()
    storage._executor = ThreadPoolExecutor(max_workers=1)
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def always_fails():
        raise OSError("still down")

    try:
        with pytest.raises(OSError, match="still down"):
            await storage._run(always_fails)
    finally:
        storage._executor.shutdown(wait=False, cancel_futures=True)
    assert sleeps == [0.5, 1.0]


async def test_run_does_not_retry_non_transient_errors(monkeypatch):
    storage = make_storage()
    storage._g = object()
    storage._executor = ThreadPoolExecutor(max_workers=1)
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def explode():
        raise ValueError("programming error")

    try:
        with pytest.raises(ValueError, match="programming error"):
            await storage._run(explode)
    finally:
        storage._executor.shutdown(wait=False, cancel_futures=True)
    assert sleeps == []


# --------------------------------------------------------------------------
# Workspace wiring
# --------------------------------------------------------------------------


def test_workspace_defaults_to_base():
    storage = GremlinStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
    )
    assert storage.workspace == "base"
    assert storage._workspace_label() == "base"


def test_workspace_accepts_constructor_argument():
    storage = GremlinStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="my_ws",
    )
    # validate_workspace rejects path separators but accepts a clean name
    # unchanged; "my_ws" survives intact
    assert storage.workspace == "my_ws"
    assert storage._workspace_label() == "my_ws"


def test_gremlin_workspace_env_overrides_constructor(monkeypatch):
    monkeypatch.setenv("GREMLIN_WORKSPACE", "env_ws")
    storage = GremlinStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="arg_ws",
    )
    assert storage.workspace == "env_ws"
    assert storage._workspace_label() == "env_ws"
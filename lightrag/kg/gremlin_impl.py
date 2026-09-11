"""Gremlin (TinkerPop) graph storage backend for LightRAG.

Connects to any TinkerPop 3 compatible Gremlin Server (TinkerGraph, JanusGraph,
Neptune, Gremlin Server, ...) through the official ``gremlinpython`` driver
using the WebSocket Gremlin protocol, and implements
:class:`~lightrag.base.BaseGraphStorage`.

Design notes
------------
* **Workspace isolation.** The workspace is stored as the vertex *label*
  (``hasLabel(workspace)``), mirroring how ``MemgraphStorage`` uses one label
  per workspace. Unlike the Cypher backends, no escaping/injection hardening
  is needed: in Gremlin bytecode the label is passed as a *data* argument to
  ``addV(...)`` / ``hasLabel(...)``, never spliced into a query string.

* **Synchronous driver in an async codebase.** ``gremlinpython``'s
  ``DriverRemoteConnection`` is synchronous (websocket-client based) but owns a
  thread-safe connection pool. Every traversal is executed inside a bounded
  :class:`~concurrent.futures.ThreadPoolExecutor`, so the class keeps the
  async interface of every other LightRAG backend while sharing the pool
  across callers.

* **Injection-free filtering.** Entity identity is the ``entity_id`` property
  on the workspace vertex (matches Neo4j / Memgraph). Counts, lookups, and
  groups are expressed with ``inject(...)`` + ``select(...).by(...)`` instead
  of Cypher ``UNWIND``.

* **Testability.** Traversal construction is factored into module-level
  ``build_*`` functions that return a plain ``gremlinpython`` traversal whose
  ``bytecode.step_instructions`` can be asserted offline against a recording
  traversal source; the storage methods only execute them.

* **Edge orientation.** LightRAG edges are semantically undirected, but each
  relation is stored once with the orientation the extraction produced.
  Listing operations return each stored edge exactly once with its stored
  ``(outV, inV)`` orientation (equivalent to Memgraph's ``startNode``
  reporting), deduplication is therefore not needed by callers.
"""

import asyncio
import configparser
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, final

import pipmaster as pm
from dotenv import load_dotenv

from ..base import BaseGraphStorage
from ..kg.shared_storage import get_data_init_lock
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger, validate_workspace

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

if not pm.is_installed("gremlinpython"):
    pm.install("gremlinpython")

from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import P

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

EDGE_LABEL = "DIRECTED"
"""Edge label used for every relation inside a workspace graph."""

# Config file fallback for connection parameters, mirroring the other backends.
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


T = TypeVar("T")


def _flatten_map_with(values: list[Any]) -> dict[str, Any]:
    """Normalize a ``valueMap().by(unfold())`` row into a plain scalar dict.

    ``valueMap().by(unfold())`` returns ``{key: scalar}`` already; this helper
    only coerces non-string values and guards against ``None`` results coming
    from a serializer that emits empty rows.
    """
    result: dict[str, Any] = {}
    for value in values:
        if value is None:
            continue
        for key, val in value.items():
            result[str(key)] = val
    return result


# --------------------------------------------------------------------------
# Traversal builders. Each returns a ``gremlinpython`` traversal (bytecode
# only, never executed) and can be asserted offline in unit tests.
# --------------------------------------------------------------------------


def build_has_node(
    g: Any, workspace: str, node_id: str
) -> Any:
    """Count lookup for a single entity."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", node_id)
        .limit(1)
        .count()
    )


def build_has_nodes(g: Any, workspace: str, node_ids: list[str]) -> Any:
    """Return the entity ids that exist, batched by ``inject``."""
    return (
        g.inject(node_ids)
        .unfold()
        .as_("id")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("id"))
        .values("entity_id")
    )


def build_node_degree(g: Any, workspace: str, node_id: str) -> Any:
    """Degree (number of incident edges) of a single entity."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", node_id)
        .bothE()
        .count()
    )


def build_node_degrees(g: Any, workspace: str, node_ids: list[str]) -> Any:
    """``{entity_id: degree}`` for all *existing* entities in ``node_ids``."""
    return (
        g.inject(node_ids)
        .unfold()
        .as_("id")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("id"))
        .group()
        .by("entity_id")
        .by(__.bothE().count())
    )


def build_has_edge(g: Any, workspace: str, source: str, target: str) -> Any:
    """Count lookup for an undirected edge between two entities."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", source)
        .bothE()
        .where(__.otherV().has("entity_id", target))
        .limit(1)
        .count()
    )


def build_get_node(g: Any, workspace: str, node_id: str) -> Any:
    """Single entity properties."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", node_id)
        .valueMap()
        .by(__.unfold())
    )


def build_get_nodes_batch(g: Any, workspace: str, node_ids: list[str]) -> Any:
    """``{entity_id: properties}`` for all *existing* entities in ``node_ids``."""
    return (
        g.inject(node_ids)
        .unfold()
        .as_("id")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("id"))
        .group()
        .by("entity_id")
        .by(__.valueMap().by(__.unfold()))
    )


def build_get_node_edges(g: Any, workspace: str, node_id: str) -> Any:
    """All directed ``(outV, inV)`` entity-id pairs touching an entity.

    The traverser is the neighbor vertex only when the entity exists; the
    caller decides "absent vs isolated" with ``has_node`` first.
    """
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", node_id)
        .bothE()
        .where(__.otherV().has("entity_id"))
        .project("s", "t")
        .by(__.outV().values("entity_id"))
        .by(__.inV().values("entity_id"))
    )


def build_get_edge(g: Any, workspace: str, source: str, target: str) -> Any:
    """Properties of the edge between two entities."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", source)
        .bothE()
        .where(__.otherV().has("entity_id", target))
        .valueMap()
        .by(__.unfold())
    )


def build_get_edges_batch(g: Any, workspace: str, pairs: list[tuple[str, str]]) -> Any:
    """Edge properties for each ``(source, target)`` pair that has an edge."""
    rows = [{"src": src, "tgt": tgt} for src, tgt in pairs]
    return (
        g.inject(rows)
        .unfold()
        .as_("row")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("src"))
        .as_("a")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("tgt"))
        .as_("b")
        .select("a")
        .bothE()
        .where(__.otherV().where(P.eq("b")))
        .project("src", "tgt", "props")
        .by(__.select("a").values("entity_id"))
        .by(__.select("b").values("entity_id"))
        .by(__.valueMap().by(__.unfold()))
    )


def build_add_node_props(g: Any, workspace: str, node_id: str, node_data: dict[str, Any]) -> Any:
    """Merge-or-create a single entity, then set every non-identity property."""
    row = dict(node_data, entity_id=node_id)
    t = (
        g.inject([row])
        .unfold()
        .as_("row")
        .coalesce(
            __.V()
            .hasLabel(workspace)
            .has("entity_id", __.select("row").by("entity_id")),
            __.addV(workspace).property(
                "entity_id", __.select("row").by("entity_id")
            ),
        )
    )
    for key in row:
        if key != "entity_id":
            t = t.property(key, __.select("row").by(key))
    return t


def build_add_nodes_batch(
    g: Any, workspace: str, nodes: list[tuple[str, dict[str, Any]]]
) -> Any:
    """Merge-or-create many entities in one injected traversal."""
    if not nodes:
        raise ValueError("build_add_nodes_batch requires at least one node")
    rows = [dict(node_data, entity_id=node_id) for node_id, node_data in nodes]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys.discard("entity_id")
    t = (
        g.inject(rows)
        .unfold()
        .as_("row")
        .coalesce(
            __.V()
            .hasLabel(workspace)
            .has("entity_id", __.select("row").by("entity_id")),
            __.addV(workspace).property(
                "entity_id", __.select("row").by("entity_id")
            ),
        )
    )
    for key in sorted(all_keys):
        t = t.property(key, __.select("row").by(key))
    return t


def build_add_edge(
    g: Any,
    workspace: str,
    source: str,
    target: str,
    edge_data: dict[str, Any],
) -> Any:
    """Merge-or-create the ``DIRECTED`` edge between two entities."""
    row = dict(edge_data, src=source, tgt=target)
    t = (
        g.inject([row])
        .unfold()
        .as_("row")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("src"))
        .as_("a")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("tgt"))
        .as_("b")
        .coalesce(
            __.select("a")
            .bothE()
            .where(__.otherV().where(P.eq("b"))),
            __.select("a").addE(EDGE_LABEL).from_("a").to("b"),
        )
    )
    for key in row:
        if key not in ("src", "tgt"):
            t = t.property(key, __.select("row").by(key))
    return t


def build_add_edges_batch(
    g: Any,
    workspace: str,
    edges: list[tuple[str, str, dict[str, Any]]],
) -> Any:
    """Merge-or-create many edges in one injected traversal."""
    if not edges:
        raise ValueError("build_add_edges_batch requires at least one edge")
    rows = [dict(edge_data, src=src, tgt=tgt) for src, tgt, edge_data in edges]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys.discard("src")
    all_keys.discard("tgt")
    t = (
        g.inject(rows)
        .unfold()
        .as_("row")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("src"))
        .as_("a")
        .V()
        .hasLabel(workspace)
        .has("entity_id", __.select("row").by("tgt"))
        .as_("b")
        .coalesce(
            __.select("a")
            .bothE()
            .where(__.otherV().where(P.eq("b"))),
            __.select("a").addE(EDGE_LABEL).from_("a").to("b"),
        )
    )
    for key in sorted(all_keys):
        t = t.property(key, __.select("row").by(key))
    return t


def build_delete_node(g: Any, workspace: str, node_id: str) -> Any:
    """DETACH-delete a single entity."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", node_id)
        .drop()
    )


def build_remove_nodes(g: Any, workspace: str, node_ids: list[str]) -> Any:
    """DETACH-delete many entities."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", P.within(node_ids))
        .drop()
    )


def build_remove_edge(g: Any, workspace: str, source: str, target: str) -> Any:
    """Delete the edge between two entities (undirected match)."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", source)
        .bothE()
        .where(__.otherV().has("entity_id", target))
        .drop()
    )


def build_drop_all(g: Any, workspace: str) -> Any:
    """Drop every vertex of the workspace (edges cascade)."""
    return g.V().hasLabel(workspace).drop()


def build_all_labels(g: Any, workspace: str) -> Any:
    """All entity ids, distinct and code-point ordered."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id")
        .values("entity_id")
        .dedup()
        .order()
    )


def build_all_nodes(g: Any, workspace: str) -> Any:
    """Properties of every vertex in the workspace."""
    return (
        g.V()
        .hasLabel(workspace)
        .valueMap()
        .by(__.unfold())
    )


def build_all_edges(g: Any, workspace: str) -> Any:
    """One row per stored edge: source, target, and properties.

    Uses ``outE`` so each stored edge is emitted exactly once with its
    stored orientation (Memgraph's ``startNode`` convention).
    """
    return (
        g.V()
        .hasLabel(workspace)
        .as_("a")
        .outE()
        .as_("r")
        .inV()
        .hasLabel(workspace)
        .project("src", "tgt", "props")
        .by(__.select("a").values("entity_id"))
        .by(__.values("entity_id"))
        .by(__.select("r").valueMap().by(__.unfold()))
    )


def build_kg_degrees(g: Any, workspace: str) -> Any:
    """``{entity_id: degree}`` for the whole workspace (isolated: degree 0)."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id")
        .group()
        .by("entity_id")
        .by(__.bothE().count())
    )


def build_kg_nodes(g: Any, workspace: str, entities: list[str]) -> Any:
    """Properties of the given entities only (used for the ``*`` view)."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", P.within(entities))
        .valueMap()
        .by(__.unfold())
    )


def build_kg_edges(g: Any, workspace: str, entities: list[str]) -> Any:
    """Stored edges whose endpoints are both within ``entities``."""
    return (
        g.V()
        .hasLabel(workspace)
        .has("entity_id", P.within(entities))
        .as_("a")
        .outE()
        .as_("r")
        .inV()
        .has("entity_id", P.within(entities))
        .project("src", "tgt", "props")
        .by(__.select("a").values("entity_id"))
        .by(__.values("entity_id"))
        .by(__.select("r").valueMap().by(__.unfold()))
    )


# --------------------------------------------------------------------------
# Storage implementation
# --------------------------------------------------------------------------


@final
@dataclass
class GremlinStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Priority: 1) GREMLIN_WORKSPACE env 2) user arg 3) default 'base'
        gremlin_workspace = os.environ.get("GREMLIN_WORKSPACE")
        original_workspace = workspace  # Save original value for logging
        if gremlin_workspace and gremlin_workspace.strip():
            workspace = gremlin_workspace

        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        validate_workspace(self.workspace)

        # Log after super().__init__() to ensure self.workspace is initialized
        if gremlin_workspace and gremlin_workspace.strip():
            logger.info(
                f"Using GREMLIN_WORKSPACE environment variable: '{gremlin_workspace}' (overriding '{original_workspace}/{namespace}')"
            )

        self._remote = None
        self._g = None
        self._executor: ThreadPoolExecutor | None = None

    def _workspace_label(self) -> str:
        """Return the workspace string used as the vertex label.

        In Gremlin bytecode the label travels as a data argument, so -- unlike
        the Cypher backends that must escape backticks -- no sanitization is
        needed. ``validate_workspace`` already rejected path separators.
        """
        workspace = self.workspace.strip()
        if not workspace:
            return "base"
        return workspace

    def _check_initialized(self) -> None:
        if self._executor is None or self._g is None:
            raise RuntimeError(
                "GremlinStorage is not initialized. Call 'await initialize()' first."
            )

    async def _run(self, thunk: Callable[[], T]) -> T:
        """Execute a traversal-building thunk in the bounded thread pool.

        Gremlin's ``DriverRemoteConnection`` is synchronous; the thread pool
        shares its internal connection pool across concurrent callers. A
        small number of connection-level retries keeps transient network
        blips from failing an ingestion batch.
        """
        self._check_initialized()
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await loop.run_in_executor(self._executor, thunk)
            except (OSError, ConnectionError, TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 0.5 * (2**attempt)
                    logger.warning(
                        f"[{self.workspace}] Gremlin connection error (attempt "
                        f"{attempt + 1}/3, retrying in {delay:.1f}s): {exc}"
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def _iterate_batches(
        self,
        traversal_factory: Callable[[], Any],
        batch_size: int,
    ):
        """Bridge a streaming Gremlin traversal into async batch yields.

        The traversal is iterated inside one executor thread; results are
        passed to the event loop through a thread-safe queue, so server-side
        streaming (``DriverRemoteConnection`` pulls results on demand) keeps
        client memory bounded to one batch at a time.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._check_initialized()
        loop = asyncio.get_running_loop()
        q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def _producer() -> None:
            try:
                t = traversal_factory()
                batch: list[Any] = []
                for item in t:
                    batch.append(item)
                    if len(batch) == batch_size:
                        q.put(("batch", batch))
                        batch = []
                if batch:
                    q.put(("batch", batch))
                q.put(("done", None))
            except Exception as exc:  # noqa: BLE001 - must cross threads
                q.put(("error", exc))

        loop.run_in_executor(self._executor, _producer)
        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "done":
                break
            if kind == "error":
                raise payload
            yield payload

    async def initialize(self):
        async with get_data_init_lock():
            URI = os.environ.get(
                "GREMLIN_URI",
                config.get("gremlin", "uri", fallback="ws://localhost:8182/gremlin"),
            )
            USERNAME = os.environ.get(
                "GREMLIN_USERNAME", config.get("gremlin", "username", fallback="")
            )
            PASSWORD = os.environ.get(
                "GREMLIN_PASSWORD", config.get("gremlin", "password", fallback="")
            )
            TRAVERSAL_SOURCE = os.environ.get(
                "GREMLIN_TRAVERSAL_SOURCE",
                config.get("gremlin", "traversal_source", fallback="g"),
            )
            POOL_SIZE = int(
                os.environ.get(
                    "GREMLIN_MAX_POOL_SIZE",
                    config.get("gremlin", "max_pool_size", fallback="8"),
                )
            )

            try:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(4, POOL_SIZE),
                    thread_name_prefix="gremlin",
                )
                self._remote = DriverRemoteConnection(
                    URI,
                    TRAVERSAL_SOURCE,
                    username=USERNAME,
                    password=PASSWORD,
                    pool_size=POOL_SIZE,
                )
                self._g = traversal().withRemote(self._remote)
                # Force a connection so an unreachable server fails fast here
                # instead of on the first data operation.
                await self._run(lambda: self._g.inject(1).count().toList())
                logger.info(
                    f"[{self.workspace}] Connected to Gremlin Server at {URI} "
                    f"(traversal source '{TRAVERSAL_SOURCE}')"
                )
            except Exception as e:
                self._g = None
                self._remote = None
                if self._executor is not None:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor = None
                logger.error(
                    f"[{self.workspace}] Failed to connect to Gremlin Server at {URI}: {e}"
                )
                raise

    async def finalize(self):
        self._g = None
        if self._remote is not None:
            try:
                await asyncio.to_thread(self._remote.close)
            except Exception:  # noqa: BLE001 - best-effort close
                pass
            self._remote = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self):
        # Gremlin Server persists everything immediately, nothing to flush.
        pass

    # ------------------------------------------------------------------
    # Existence / degree
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        t = build_has_node(self._g, self._workspace_label(), node_id)
        result = await self._run(lambda t=t: t.toList())
        return bool(result and result[0] > 0)

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        if not node_ids:
            return set()
        t = build_has_nodes(self._g, self._workspace_label(), node_ids)
        result = await self._run(lambda t=t: t.toList())
        return set(result)

    async def node_degree(self, node_id: str) -> int:
        t = build_node_degree(self._g, self._workspace_label(), node_id)
        result = await self._run(lambda t=t: t.toList())
        if not result:
            return 0
        return int(result[0])

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        if not node_ids:
            return result
        t = build_node_degrees(self._g, self._workspace_label(), node_ids)
        degrees = await self._run(lambda t=t: t.toList())
        found: dict[str, int] = {}
        for row in degrees:
            if row is None:
                continue
            for entity_id, degree in row.items():
                found[str(entity_id)] = int(degree)
        for node_id in node_ids:
            result[node_id] = found.get(node_id, 0)
        return result

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        t = build_has_edge(
            self._g, self._workspace_label(), source_node_id, target_node_id
        )
        result = await self._run(lambda t=t: t.toList())
        return bool(result and result[0] > 0)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (await self.node_degree(src_id)) + (await self.node_degree(tgt_id))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        t = build_get_node(self._g, self._workspace_label(), node_id)
        result = await self._run(lambda t=t: t.toList())
        if not result:
            return None
        props = _flatten_map_with(result[:1])
        if not props:
            return None
        return props

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if not node_ids:
            return result
        t = build_get_nodes_batch(self._g, self._workspace_label(), node_ids)
        groups = await self._run(lambda t=t: t.toList())
        for row in groups:
            if row is None:
                continue
            for entity_id, props in row.items():
                result[str(entity_id)] = _flatten_map_with([props])
        return result

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        t = build_get_edge(
            self._g, self._workspace_label(), source_node_id, target_node_id
        )
        result = await self._run(lambda t=t: t.toList())
        if not result:
            return None
        edge_result = _flatten_map_with(result[:1])
        if not edge_result:
            return None
        for key, default_value in {
            "weight": 1.0,
            "source_id": None,
            "description": None,
            "keywords": None,
        }.items():
            if key not in edge_result:
                edge_result[key] = default_value
                logger.warning(
                    f"[{self.workspace}] Edge between {source_node_id} and "
                    f"{target_node_id} is missing property: {key}. Using "
                    f"default value: {default_value}"
                )
        return edge_result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        result: dict[tuple[str, str], dict] = {}
        if not pairs:
            return result
        flat = [(pair["src"], pair["tgt"]) for pair in pairs]
        t = build_get_edges_batch(self._g, self._workspace_label(), flat)
        rows = await self._run(lambda t=t: t.toList())
        for row in rows:
            if row is None:
                continue
            src = str(row.get("src", ""))
            tgt = str(row.get("tgt", ""))
            if src and tgt:
                result[(src, tgt)] = _flatten_map_with([row.get("props")])
        return result

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Return directed ``(outV, inV)`` pairs touching ``source_node_id``.

        ``[]`` -- the node exists and has no relations. ``None`` -- the node
        is confirmed absent. A backend error raises. (BaseGraphStorage
        three-outcome contract.)
        """
        if not await self.has_node(source_node_id):
            return None
        t = build_get_node_edges(self._g, self._workspace_label(), source_node_id)
        rows = await self._run(lambda t=t: t.toList())
        edges: list[tuple[str, str]] = []
        for row in rows:
            if row is None:
                continue
            src = row.get("s")
            tgt = row.get("t")
            if src is not None and tgt is not None:
                edges.append((str(src), str(tgt)))
        return edges

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        t = build_add_node_props(self._g, self._workspace_label(), node_id, node_data)
        await self._run(lambda t=t: t.iterate())

    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        if not nodes:
            return
        t = build_add_nodes_batch(self._g, self._workspace_label(), nodes)
        await self._run(lambda t=t: t.iterate())

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        t = build_add_edge(
            self._g,
            self._workspace_label(),
            source_node_id,
            target_node_id,
            edge_data,
        )
        await self._run(lambda t=t: t.iterate())

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        if not edges:
            return
        t = build_add_edges_batch(self._g, self._workspace_label(), edges)
        await self._run(lambda t=t: t.iterate())

    async def delete_node(self, node_id: str) -> None:
        t = build_delete_node(self._g, self._workspace_label(), node_id)
        await self._run(lambda t=t: t.iterate())

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        t = build_remove_nodes(self._g, self._workspace_label(), nodes)
        await self._run(lambda t=t: t.iterate())

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        for source, target in edges:
            t = build_remove_edge(self._g, self._workspace_label(), source, target)
            await self._run(lambda t=t: t.iterate())

    async def drop(self) -> dict[str, str]:
        try:
            t = build_drop_all(self._g, self._workspace_label())
            await self._run(lambda t=t: t.iterate())
            logger.info(
                f"[{self.workspace}] Dropped workspace data from Gremlin Server"
            )
            return {"status": "success", "message": "workspace data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping workspace data from Gremlin Server: {e}"
            )
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Whole-graph iteration
    # ------------------------------------------------------------------

    async def get_all_labels(self) -> list[str]:
        t = build_all_labels(self._g, self._workspace_label())
        return [str(x) for x in await self._run(lambda t=t: t.toList())]

    async def iter_labels(self, batch_size: int):
        ws = self._workspace_label()
        async for batch in self._iterate_batches(
            lambda: build_all_labels(self._g, ws), batch_size
        ):
            yield [str(x) for x in batch]

    async def get_all_nodes(self) -> list[dict]:
        t = build_all_nodes(self._g, self._workspace_label())
        rows = await self._run(lambda t=t: t.toList())
        nodes = []
        for row in rows:
            props = _flatten_map_with([row])
            if not props:
                continue
            props["id"] = props.get("entity_id")
            nodes.append(props)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        t = build_all_edges(self._g, self._workspace_label())
        rows = await self._run(lambda t=t: t.toList())
        edges = []
        for row in rows:
            if row is None:
                continue
            edge_properties = _flatten_map_with([row.get("props")])
            edge_properties["source"] = row.get("src")
            edge_properties["target"] = row.get("tgt")
            edges.append(edge_properties)
        return edges

    async def iter_edges(self, batch_size: int):
        ws = self._workspace_label()
        async for batch in self._iterate_batches(
            lambda: build_all_edges(self._g, ws), batch_size
        ):
            out = []
            for row in batch:
                if row is None:
                    continue
                edge_properties = _flatten_map_with([row.get("props")])
                edge_properties["source"] = row.get("src")
                edge_properties["target"] = row.get("tgt")
                out.append(edge_properties)
            yield out

    # ------------------------------------------------------------------
    # Ranking / search
    # ------------------------------------------------------------------

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Ranks the whole node set by degree (highest first); ties break on
        the label ascending -- the BaseGraphStorage contract.

        Gremlin has no server-side ORDER BY over the grouped projection, so the
        ranking is materialized and sorted client-side. The full-pass
        alternative explicitly permitted by the contract is used: the group
        query surfaces every entity including isolated ones (degree 0), so one
        round trip answers the whole rank. A backend error raises rather than
        returning [] (empty list is reserved for "the graph has no entities").
        """
        t = build_kg_degrees(self._g, self._workspace_label())
        groups = await self._run(lambda t=t: t.toList())
        degrees: dict[str, int] = {}
        for row in groups:
            if row is None:
                continue
            for entity_id, degree in row.items():
                degrees[str(entity_id)] = int(degree)
        ranked = sorted(degrees.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [label for label, _ in ranked]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Fuzzy entity search, Memgraph-compatible scoring.

        Exact match scores highest, then prefix, then substring by remaining
        length. Matching and scoring happen client-side for case-insensitive
        semantics that only a handful of providers support server-side. A
        blank query is a real empty result, not an error.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        t = build_all_labels(self._g, self._workspace_label())
        labels = [str(x) for x in await self._run(lambda t=t: t.toList())]

        scored = []
        for label in labels:
            label_lower = label.lower()
            if query_lower not in label_lower:
                continue
            if label_lower == query_lower:
                score = 1000
            elif label_lower.startswith(query_lower):
                score = 500
            else:
                score = 100 - len(label_lower)
            scored.append((label, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [label for label, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(
                max_nodes, self.global_config.get("max_graph_nodes", 1000)
            )

        if node_label == "*":
            return await self._kg_whole_graph(max_nodes)
        return await self._kg_subgraph(node_label, max_depth, max_nodes)

    async def _kg_whole_graph(self, max_nodes: int) -> KnowledgeGraph:
        """Rank the whole node set, then fetch kept nodes and internal edges."""
        t = build_kg_degrees(self._g, self._workspace_label())
        groups = await self._run(lambda t=t: t.toList())
        degrees: dict[str, int] = {}
        for row in groups:
            if row is None:
                continue
            for entity_id, degree in row.items():
                degrees[str(entity_id)] = int(degree)

        # Degree descending, entity_id ascending: the BaseGraphStorage
        # tie-break contract for the truncated view.
        ranked = sorted(degrees.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [label for label, _ in ranked[:max_nodes]]
        result = KnowledgeGraph(
            is_truncated=len(degrees) > max_nodes,
        )

        if not kept:
            return result

        t_nodes = build_kg_nodes(self._g, self._workspace_label(), kept)
        rows = await self._run(lambda t=t_nodes: t.toList())
        for row in rows:
            props = _flatten_map_with([row])
            entity_id = props.get("entity_id")
            if entity_id is None:
                continue
            entity_id = str(entity_id)
            result.nodes.append(
                KnowledgeGraphNode(
                    id=entity_id,
                    labels=[entity_id],
                    properties=props,
                )
            )

        t_edges = build_kg_edges(self._g, self._workspace_label(), kept)
        rows = await self._run(lambda t=t_edges: t.toList())
        for row in rows:
            if row is None:
                continue
            src = row.get("src")
            tgt = row.get("tgt")
            if src is None or tgt is None:
                continue
            src, tgt = str(src), str(tgt)
            result.edges.append(
                KnowledgeGraphEdge(
                    id=f"{src}->{tgt}",
                    type=EDGE_LABEL,
                    source=src,
                    target=tgt,
                    properties=_flatten_map_with([row.get("props")]),
                )
            )
        return result

    async def _kg_subgraph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """BFS expansion from a start entity (exact ``entity_id`` match),
        bounded by ``max_depth`` then truncated by ``max_nodes`` keeping the
        start node first (Memgraph behavior)."""
        result = KnowledgeGraph()
        if not await self.has_node(node_label):
            return result

        discovered: list[str] = [node_label]
        visited: set[str] = {node_label}
        frontier: list[str] = [node_label]
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for current in frontier:
                edges = await self.get_node_edges(current)
                if edges is None:
                    continue
                for _, neighbor in edges:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            if not next_frontier:
                break
            next_frontier.sort()
            discovered.extend(next_frontier)
            frontier = next_frontier
            if len(discovered) >= max_nodes:
                break

        truncated = len(discovered) > max_nodes
        kept = discovered[:max_nodes]
        result.is_truncated = truncated

        props_map = await self.get_nodes_batch(kept)
        for entity_id in kept:
            props = props_map.get(entity_id, {"entity_id": entity_id})
            result.nodes.append(
                KnowledgeGraphNode(
                    id=entity_id,
                    labels=[entity_id],
                    properties=props,
                )
            )

        seen_edges: set[frozenset[str]] = set()
        kept_set = set(kept)
        pairs: list[tuple[str, str]] = []
        for entity_id in kept:
            edges = await self.get_node_edges(entity_id)
            if edges is None:
                continue
            for src, tgt in edges:
                if src in kept_set and tgt in kept_set:
                    pair_key = frozenset((src, tgt))
                    if pair_key not in seen_edges:
                        seen_edges.add(pair_key)
                        pairs.append((src, tgt))

        if pairs:
            fetched = await self.get_edges_batch(
                [{"src": src, "tgt": tgt} for src, tgt in pairs]
            )
        else:
            fetched = {}
        for src, tgt in pairs:
            props = fetched.get((src, tgt), {})
            result.edges.append(
                KnowledgeGraphEdge(
                    id=f"{src}->{tgt}",
                    type=EDGE_LABEL,
                    source=src,
                    target=tgt,
                    properties=props,
                )
            )
        return result
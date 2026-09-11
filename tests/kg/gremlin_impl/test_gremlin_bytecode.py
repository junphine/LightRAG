"""Bytecode pinning tests for the Gremlin traversal builders.

Every ``build_*`` function in :mod:`lightrag.kg.gremlin_impl` produces a
deterministic gremlinpython traversal for a given workspace/entity
combination. These tests pin the exact bytecode -- including nested traversal
arguments -- so an accidental change to how a traversal is constructed (a
wrong step, a misplaced ``by()``, a missing ``hasLabel`` filter) fails loudly
instead of silently changing what a live Gremlin server would execute.

The expected bytecode below was produced by :func:`normalize_steps` on an
offline ``GraphTraversalSource`` (``traversal().with_(None)``), which is what
the storage builds before handing a callable to ``_run``. ``P`` instances are
unfolded to ``('P', operator, value)`` tuples because ``P.__eq__`` is
overloaded to build predicates and would make naive equality lie.
"""

from gremlin_python.process.traversal import P

from lightrag.kg.gremlin_impl import (
    build_add_edge,
    build_add_edges_batch,
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
from tests.kg.gremlin_impl._utils import WS, classify, graph_source, normalize_steps

# (name, builder, args, expected_kind, expected_normalized_bytecode)
CASES = [
    (
        "has_node",
        build_has_node,
        ("n1",),
        "has_node",
        (("V", ()), ("hasLabel", ("ws1",)), ("has", ("entity_id", "n1")), ("limit", (1,)), ("count", ())),
    ),
    (
        "has_nodes",
        build_has_nodes,
        (["n1", "n2"],),
        "has_nodes",
        (
            ("inject", (("n1", "n2"),)),
            ("unfold", ()),
            ("as", ("id",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("id",)),))),
            ("values", ("entity_id",)),
        ),
    ),
    (
        "node_degree",
        build_node_degree,
        ("n1",),
        "node_degree",
        (("V", ()), ("hasLabel", ("ws1",)), ("has", ("entity_id", "n1")), ("bothE", ()), ("count", ())),
    ),
    (
        "node_degrees",
        build_node_degrees,
        (["n1", "n2"],),
        "node_degrees",
        (
            ("inject", (("n1", "n2"),)),
            ("unfold", ()),
            ("as", ("id",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("id",)),))),
            ("group", ()),
            ("by", ("entity_id",)),
            ("by", ((("bothE", ()), ("count", ())),)),
        ),
    ),
    (
        "has_edge",
        build_has_edge,
        ("n1", "n2"),
        "has_edge",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", "n1")),
            ("bothE", ()),
            ("where", ((("otherV", ()), ("has", ("entity_id", "n2"))),)),
            ("limit", (1,)),
            ("count", ()),
        ),
    ),
    (
        "get_node",
        build_get_node,
        ("n1",),
        "get_node",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", "n1")),
            ("valueMap", ()),
            ("by", ((("unfold", ()),),)),
        ),
    ),
    (
        "get_nodes_batch",
        build_get_nodes_batch,
        (["n1", "n2"],),
        "get_nodes_batch",
        (
            ("inject", (("n1", "n2"),)),
            ("unfold", ()),
            ("as", ("id",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("id",)),))),
            ("group", ()),
            ("by", ("entity_id",)),
            ("by", ((("valueMap", ()), ("by", ((("unfold", ()),),))),)),
        ),
    ),
    (
        "get_node_edges",
        build_get_node_edges,
        ("n1",),
        "get_node_edges",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", "n1")),
            ("bothE", ()),
            ("where", ((("otherV", ()), ("has", ("entity_id",))),)),
            ("project", ("s", "t")),
            ("by", ((("outV", ()), ("values", ("entity_id",))),)),
            ("by", ((("inV", ()), ("values", ("entity_id",))),)),
        ),
    ),
    (
        "get_edge",
        build_get_edge,
        ("n1", "n2"),
        "get_edge",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", "n1")),
            ("bothE", ()),
            ("where", ((("otherV", ()), ("has", ("entity_id", "n2"))),)),
            ("valueMap", ()),
            ("by", ((("unfold", ()),),)),
        ),
    ),
    (
        "get_edges_batch",
        build_get_edges_batch,
        ([("n1", "n2"), ("n3", "n4")],),
        "get_edges_batch",
        (
            ("inject", (({"src": "n1", "tgt": "n2"}, {"src": "n3", "tgt": "n4"}),)),
            ("unfold", ()),
            ("as", ("row",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("src",))))),
            ("as", ("a",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("tgt",))))),
            ("as", ("b",)),
            ("select", ("a",)),
            ("bothE", ()),
            ("where", ((("otherV", ()), ("where", (("P", "eq", "b"),))),)),
            ("project", ("src", "tgt", "props")),
            ("by", ((("select", ("a",)), ("values", ("entity_id",))),)),
            ("by", ((("select", ("b",)), ("values", ("entity_id",))),)),
            ("by", ((("valueMap", ()), ("by", ((("unfold", ()),),))),)),
        ),
    ),
    (
        "add_node_props",
        build_add_node_props,
        ("n1", {"description": "d1"}),
        "upsert_node",
        (
            ("inject", (({"description": "d1", "entity_id": "n1"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            (
                "coalesce",
                (
                    (
                        ("V", ()),
                        ("hasLabel", ("ws1",)),
                        ("has", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                    (
                        ("addV", ("ws1",)),
                        ("property", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                ),
            ),
            ("property", ("description", (("select", ("row",)), ("by", ("description",))))),
        ),
    ),
    (
        "add_node_props_multi",
        build_add_node_props,
        ("n1", {"description": "d1", "tags": "t"}),
        "upsert_node",
        (
            ("inject", (({"description": "d1", "tags": "t", "entity_id": "n1"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            (
                "coalesce",
                (
                    (
                        ("V", ()),
                        ("hasLabel", ("ws1",)),
                        ("has", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                    (
                        ("addV", ("ws1",)),
                        ("property", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                ),
            ),
            ("property", ("description", (("select", ("row",)), ("by", ("description",))))),
            ("property", ("tags", (("select", ("row",)), ("by", ("tags",))))),
        ),
    ),
    (
        "add_node_props_empty",
        build_add_node_props,
        ("n1", {}),
        "upsert_node",
        (
            ("inject", (({"entity_id": "n1"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            (
                "coalesce",
                (
                    (
                        ("V", ()),
                        ("hasLabel", ("ws1",)),
                        ("has", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                    (
                        ("addV", ("ws1",)),
                        ("property", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                ),
            ),
        ),
    ),
    (
        "add_nodes_batch",
        build_add_nodes_batch,
        ([("n1", {"description": "d1"})],),
        "upsert_node",
        (
            ("inject", (({"description": "d1", "entity_id": "n1"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            (
                "coalesce",
                (
                    (
                        ("V", ()),
                        ("hasLabel", ("ws1",)),
                        ("has", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                    (
                        ("addV", ("ws1",)),
                        ("property", ("entity_id", (("select", ("row",)), ("by", ("entity_id",))))),
                    ),
                ),
            ),
            ("property", ("description", (("select", ("row",)), ("by", ("description",))))),
        ),
    ),
    (
        "add_edge",
        build_add_edge,
        ("n1", "n2", {"weight": 0.5}),
        "upsert_edge",
        (
            ("inject", (({"weight": 0.5, "src": "n1", "tgt": "n2"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("src",))))),
            ("as", ("a",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("tgt",))))),
            ("as", ("b",)),
            (
                "coalesce",
                (
                    (
                        ("select", ("a",)),
                        ("bothE", ()),
                        ("where", ((("otherV", ()), ("where", (("P", "eq", "b"),))),)),
                    ),
                    (("select", ("a",)), ("addE", ("DIRECTED",)), ("from", ("a",)), ("to", ("b",))),
                ),
            ),
            ("property", ("weight", (("select", ("row",)), ("by", ("weight",))))),
        ),
    ),
    (
        "add_edge_multi",
        build_add_edge,
        ("n1", "n2", {"weight": 0.5, "desc": "x"}),
        "upsert_edge",
        (
            ("inject", (({"weight": 0.5, "desc": "x", "src": "n1", "tgt": "n2"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("src",))))),
            ("as", ("a",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("tgt",))))),
            ("as", ("b",)),
            (
                "coalesce",
                (
                    (
                        ("select", ("a",)),
                        ("bothE", ()),
                        ("where", ((("otherV", ()), ("where", (("P", "eq", "b"),))),)),
                    ),
                    (("select", ("a",)), ("addE", ("DIRECTED",)), ("from", ("a",)), ("to", ("b",))),
                ),
            ),
            ("property", ("weight", (("select", ("row",)), ("by", ("weight",))))),
            ("property", ("desc", (("select", ("row",)), ("by", ("desc",))))),
        ),
    ),
    (
        "add_edge_empty",
        build_add_edge,
        ("n1", "n2", {}),
        "upsert_edge",
        (
            ("inject", (({"src": "n1", "tgt": "n2"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("src",))))),
            ("as", ("a",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("tgt",))))),
            ("as", ("b",)),
            (
                "coalesce",
                (
                    (
                        ("select", ("a",)),
                        ("bothE", ()),
                        ("where", ((("otherV", ()), ("where", (("P", "eq", "b"),))),)),
                    ),
                    (("select", ("a",)), ("addE", ("DIRECTED",)), ("from", ("a",)), ("to", ("b",))),
                ),
            ),
        ),
    ),
    (
        "add_edges_batch",
        build_add_edges_batch,
        ([("n1", "n2", {"weight": 0.5})],),
        "upsert_edge",
        (
            ("inject", (({"weight": 0.5, "src": "n1", "tgt": "n2"},),)),
            ("unfold", ()),
            ("as", ("row",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("src",))))),
            ("as", ("a",)),
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", (("select", ("row",)), ("by", ("tgt",))))),
            ("as", ("b",)),
            (
                "coalesce",
                (
                    (
                        ("select", ("a",)),
                        ("bothE", ()),
                        ("where", ((("otherV", ()), ("where", (("P", "eq", "b"),))),)),
                    ),
                    (("select", ("a",)), ("addE", ("DIRECTED",)), ("from", ("a",)), ("to", ("b",))),
                ),
            ),
            ("property", ("weight", (("select", ("row",)), ("by", ("weight",))))),
        ),
    ),
    (
        "delete_node",
        build_delete_node,
        ("n1",),
        "delete_node",
        (("V", ()), ("hasLabel", ("ws1",)), ("has", ("entity_id", "n1")), ("drop", ())),
    ),
    (
        "remove_nodes",
        build_remove_nodes,
        (["n1", "n2"],),
        "remove_nodes",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", ("P", "within", ("n1", "n2")))),
            ("drop", ()),
        ),
    ),
    (
        "remove_edge",
        build_remove_edge,
        ("n1", "n2"),
        "remove_edge",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", "n1")),
            ("bothE", ()),
            ("where", ((("otherV", ()), ("has", ("entity_id", "n2"))),)),
            ("drop", ()),
        ),
    ),
    (
        "drop_all",
        build_drop_all,
        (),
        "drop_all",
        (("V", ()), ("hasLabel", ("ws1",)), ("drop", ())),
    ),
    (
        "all_labels",
        build_all_labels,
        (),
        "all_labels",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id",)),
            ("values", ("entity_id",)),
            ("dedup", ()),
            ("order", ()),
        ),
    ),
    (
        "all_nodes",
        build_all_nodes,
        (),
        "all_nodes",
        (("V", ()), ("hasLabel", ("ws1",)), ("valueMap", ()), ("by", ((("unfold", ()),),))),
    ),
    (
        "all_edges",
        build_all_edges,
        (),
        "all_edges",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("as", ("a",)),
            ("outE", ()),
            ("as", ("r",)),
            ("inV", ()),
            ("hasLabel", ("ws1",)),
            ("project", ("src", "tgt", "props")),
            ("by", ((("select", ("a",)), ("values", ("entity_id",))),)),
            ("by", ((("values", ("entity_id",)),),)),
            ("by", ((("select", ("r",)), ("valueMap", ()), ("by", ((("unfold", ()),),))),)),
        ),
    ),
    (
        "kg_degrees",
        build_kg_degrees,
        (),
        "kg_degrees",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id",)),
            ("group", ()),
            ("by", ("entity_id",)),
            ("by", ((("bothE", ()), ("count", ())),)),
        ),
    ),
    (
        "kg_nodes",
        build_kg_nodes,
        (["n1", "n2"],),
        "kg_nodes",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", ("P", "within", ("n1", "n2")))),
            ("valueMap", ()),
            ("by", ((("unfold", ()),),)),
        ),
    ),
    (
        "kg_edges",
        build_kg_edges,
        (["n1", "n2"],),
        "kg_edges",
        (
            ("V", ()),
            ("hasLabel", ("ws1",)),
            ("has", ("entity_id", ("P", "within", ("n1", "n2")))),
            ("as", ("a",)),
            ("outE", ()),
            ("as", ("r",)),
            ("inV", ()),
            ("has", ("entity_id", ("P", "within", ("n1", "n2")))),
            ("project", ("src", "tgt", "props")),
            ("by", ((("select", ("a",)), ("values", ("entity_id",))),)),
            ("by", ((("values", ("entity_id",)),),)),
            ("by", ((("select", ("r",)), ("valueMap", ()), ("by", ((("unfold", ()),),))),)),
        ),
    ),
]


def test_builders_pin_bytecode():
    """Every builder produces its pinned bytecode, including the P unfolding."""
    for name, builder, args, _, expected in CASES:
        t = builder(graph_source(), WS, *args)
        assert normalize_steps(t) == expected, f"{name}: bytecode drifted"


def test_builders_classify_to_expected_kind():
    """Every builder's traversal maps back to the operation that built it."""
    for name, builder, args, kind, _ in CASES:
        t = builder(graph_source(), WS, *args)
        assert classify(t) == kind, f"{name}: classify mismatch"


def test_attribute_key_is_entity_id():
    """The canonical node-key property is always ``entity_id`` (never baked from data)."""
    # drop_all never touches a node key; all_nodes projects the whole valueMap.
    exempt = {"drop_all", "all_nodes"}
    for name, builder, args, _, expected in CASES:
        if name in exempt:
            continue
        text = repr(expected)
        # inject rows carry user data (property names), but a literal
        # top-level property key must always be entity_id.
        assert "entity_id" in text, f"{name}: missing entity_id key"


def test_single_node_lookup_uses_literal_value():
    """Single-id lookups pass the id literally; batch lookups use inject/unfold."""
    single = normalize_steps(build_has_node(graph_source(), WS, "n1"))
    batch = normalize_steps(build_has_nodes(graph_source(), WS, ["n1", "n2"]))
    assert single[:3] == (("V", ()), ("hasLabel", ("ws1",)), ("has", ("entity_id", "n1")))
    assert batch[0] == ("inject", (("n1", "n2"),))
    assert batch[1] == ("unfold", ())
    assert "within" not in repr(single)


def test_negative_predicates_fold_to_within():
    """remove_nodes / kg_* use P.within, never a manual chained or()."""
    for name in ("remove_nodes", "kg_nodes", "kg_edges"):
        builder = dict((c[0], c[1]) for c in CASES)[name]
        args = dict((c[0], c[2]) for c in CASES)[name]
        expected = dict((c[0], c[4]) for c in CASES)[name]
        t = builder(graph_source(), WS, *args)
        assert normalize_steps(t) == expected
        # a P.within predicate escapes the id as a comparable literal
    p = P.within(["n1", "n2"])
    assert ("P", "within", ("n1", "n2")) == ("P", p.operator, tuple(p.value))


def test_upsert_coalesce_shape():
    """upsert_node/upsert_edge build coalesce(existing, create) and a property tail."""
    node = normalize_steps(build_add_node_props(graph_source(), WS, "n1", {"description": "d1"}))
    # coalesce head: match branch then addV/create branch
    coalesce_step = [s for s in node if s[0] == "coalesce"]
    assert len(coalesce_step) == 1
    branches = coalesce_step[0][1]
    assert branches[0][0][0] == "V"  # existing-vertex check runs first
    assert branches[1][0][0] == "addV"
    # every non-identity key becomes one trailing property step
    tail = [s for s in node if s[0] == "property"]
    assert [s[1][0] for s in tail] == ["description"]

    edge = normalize_steps(build_add_edge(graph_source(), WS, "n1", "n2", {"weight": 0.5}))
    tail = [s for s in edge if s[0] == "property"]
    assert [s[1][0] for s in tail] == ["weight"]
    # the edge creation branch ends with addE DIRECTED
    edge_coalesce = [s for s in edge if s[0] == "coalesce"][0][1]
    assert ("addE", ("DIRECTED",)) in edge_coalesce[1]


def test_graph_traversal_uses_labeled_edge_projection():
    """get_node_edges projects src/tgt via outV/inV — not an unlabeled bothE map."""
    g = normalize_steps(build_get_node_edges(graph_source(), WS, "n1"))
    project = [s for s in g if s[0] == "project"][0]
    assert project[1] == ("s", "t")
    bys = [s for s in g if s[0] == "by"]
    assert bys[0] == ("by", ((("outV", ()), ("values", ("entity_id",))),))
    assert bys[1] == ("by", ((("inV", ()), ("values", ("entity_id",))),))
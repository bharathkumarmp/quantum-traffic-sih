from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.algorithms.route_utils import RouteUtils


def build_graph():
    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")

    graph = Graph()

    graph.add_edge(
        Edge(
            n0,
            n1,
            distance=10,
            travel_time=10,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            n0,
            n2,
            distance=5,
            travel_time=5,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            n1,
            n3,
            distance=10,
            travel_time=10,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            n2,
            n3,
            distance=5,
            travel_time=5,
            congestion=0.1,
        )
    )

    return graph


def test_score_to_index():
    assert (
        RouteUtils.edge_score_to_index(
            0.0,
            2
        )
        == 0
    )

    assert (
        RouteUtils.edge_score_to_index(
            0.99,
            2
        )
        == 1
    )


def test_build_route():
    graph = build_graph()

    route = RouteUtils.build_route(
        graph,
        "N0",
        "N3",
        [0.99, 0.0],
    )

    assert route is not None

    assert [node.id for node in route] == [
        "N0",
        "N2",
        "N3",
    ]


def test_route_validation():
    graph = build_graph()

    route = RouteUtils.build_route(
        graph,
        "N0",
        "N3",
        [0.99, 0.0],
    )

    assert RouteUtils.route_is_valid(
        graph,
        route,
        "N0",
        "N3",
    )


def test_blocked_edge():
    graph = build_graph()

    graph.get_edges()[1].blocked = True

    route = RouteUtils.build_route(
        graph,
        "N0",
        "N3",
        [0.99, 0.0],
    )

    assert route is not None

    assert [node.id for node in route] == [
        "N0",
        "N1",
        "N3",
    ]

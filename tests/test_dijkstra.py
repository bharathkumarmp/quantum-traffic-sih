from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph
from src.algorithms.dijkstra import Dijkstra


def build_graph():
    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")

    graph = Graph()

    # Long route
    graph.add_edge(
        Edge(
            n0,
            n1,
            distance=10,
            travel_time=10,
            congestion=0.2,
        )
    )

    graph.add_edge(
        Edge(
            n1,
            n3,
            distance=10,
            travel_time=10,
            congestion=0.2,
        )
    )

    # Short route
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
            n2,
            n3,
            distance=5,
            travel_time=5,
            congestion=0.1,
        )
    )

    return graph


def test_dijkstra_finds_shortest_route():
    graph = build_graph()

    optimizer = Dijkstra(graph)

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    route = result["route"]

    assert [
        node.id
        for node in route
    ] == [
        "N0",
        "N2",
        "N3",
    ]


def test_dijkstra_travel_time():
    graph = build_graph()

    optimizer = Dijkstra(graph)

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert result["travel_time"] == 10


def test_dijkstra_distance():
    graph = build_graph()

    optimizer = Dijkstra(graph)

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert result["distance"] == 10


def test_dijkstra_avoids_blocked_edge():
    graph = build_graph()

    # Block the short N0 -> N2 edge.
    #
    # Edge order is:
    # 0 = N0 -> N1
    # 1 = N1 -> N3
    # 2 = N0 -> N2
    # 3 = N2 -> N3
    #
    # Therefore the N0 -> N2 edge is index 2.
    graph.get_edges()[2].blocked = True

    optimizer = Dijkstra(graph)

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    route = result["route"]

    assert [
        node.id
        for node in route
    ] == [
        "N0",
        "N1",
        "N3",
    ]
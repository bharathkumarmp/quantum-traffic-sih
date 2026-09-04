from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.algorithms.pso import PSO


def build_graph():

    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")

    graph = Graph()

    # Longer route
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

    # Shorter route
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


def test_pso_initialization():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=10,
        iterations=10,
        seed=42,
    )

    assert optimizer.particle_count == 10
    assert optimizer.iterations == 10
    assert optimizer.dimension == 3


def test_pso_position():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=10,
        iterations=10,
        seed=42,
    )

    position = (
        optimizer._random_position()
    )

    assert len(position) == 3

    assert all(
        0.0 <= value <= 1.0
        for value in position
    )


def test_pso_velocity():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=10,
        iterations=10,
        seed=42,
    )

    velocity = (
        optimizer._random_velocity()
    )

    assert len(velocity) == 3


def test_pso_returns_result():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=20,
        iterations=30,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert result["route"] is not None

    assert (
        result["fitness"]
        < float("inf")
    )

    assert "travel_time" in result
    assert "distance" in result
    assert "congestion" in result
    assert "history" in result


def test_pso_route_is_valid():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=20,
        iterations=30,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert (
        result["route"][0].id
        == "N0"
    )

    assert (
        result["route"][-1].id
        == "N3"
    )


def test_pso_history():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=10,
        iterations=20,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert len(
        result["history"]
    ) == 20


def test_pso_metrics():

    graph = build_graph()

    optimizer = PSO(
        graph,
        particles=20,
        iterations=30,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert (
        result["travel_time"] > 0
    )

    assert (
        result["distance"] > 0
    )

    assert (
        0.0
        <= result["congestion"]
        <= 1.0
    )


def test_pso_reproducibility():

    graph1 = build_graph()
    graph2 = build_graph()

    optimizer1 = PSO(
        graph1,
        particles=20,
        iterations=30,
        seed=42,
    )

    optimizer2 = PSO(
        graph2,
        particles=20,
        iterations=30,
        seed=42,
    )

    result1 = optimizer1.optimize(
        "N0",
        "N3",
    )

    result2 = optimizer2.optimize(
        "N0",
        "N3",
    )

    assert (
        result1["fitness"]
        == result2["fitness"]
    )

    assert (
        [
            node.id
            for node in result1["route"]
        ]
        ==
        [
            node.id
            for node in result2["route"]
        ]
    )

    assert (
        result1["history"]
        == result2["history"]
    )
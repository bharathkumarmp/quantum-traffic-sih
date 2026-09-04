from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.algorithms.qpso import QPSO


def build_graph():
    """
    Create a small test network.

    N0 -> N1 -> N3
           |
    N0 -> N2 -> N3

    The N0 -> N2 -> N3 path is shorter.
    """

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


# =============================================================
# INITIALIZATION
# =============================================================

def test_qpso_initialization():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=10,
        iterations=10,
        beta=0.7,
        seed=42,
    )

    assert optimizer.particle_count == 10

    assert optimizer.iterations == 10

    assert optimizer.beta == 0.7

    assert optimizer.dimension == 3


# =============================================================
# RANDOM POSITION
# =============================================================

def test_qpso_position_dimension():

    graph = build_graph()

    optimizer = QPSO(
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


# =============================================================
# POSITION UPDATE
# =============================================================

def test_qpso_position_update():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=10,
        iterations=10,
        beta=0.7,
        seed=42,
    )

    position = [
        0.2,
        0.5,
        0.8,
    ]

    best_position = [
        0.3,
        0.4,
        0.7,
    ]

    mean_best = [
        0.4,
        0.5,
        0.6,
    ]

    new_position = (
        optimizer._update_position(
            position,
            best_position,
            mean_best,
        )
    )

    assert len(
        new_position
    ) == 3

    assert all(
        0.0 <= value <= 1.0
        for value in new_position
    )


# =============================================================
# QPSO RESULT
# =============================================================

def test_qpso_returns_result():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=20,
        iterations=30,
        beta=0.7,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    assert "route" in result

    assert "fitness" in result

    assert "travel_time" in result

    assert "distance" in result

    assert "congestion" in result

    assert "position" in result

    assert "history" in result

    assert "iterations" in result

    assert "particles" in result

    assert "beta" in result

    assert result["route"] is not None

    assert (
        result["fitness"]
        < float("inf")
    )


# =============================================================
# ROUTE VALIDITY
# =============================================================

def test_qpso_route_is_valid():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=20,
        iterations=30,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    route = result["route"]

    assert route is not None

    assert route[0].id == "N0"

    assert route[-1].id == "N3"


# =============================================================
# HISTORY
# =============================================================

def test_qpso_history():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=10,
        iterations=20,
        seed=42,
    )

    result = optimizer.optimize(
        "N0",
        "N3",
    )

    # One fitness value is recorded
    # after every optimization iteration.
    assert len(
        result["history"]
    ) == 20


# =============================================================
# METRICS
# =============================================================

def test_qpso_metrics_are_valid():

    graph = build_graph()

    optimizer = QPSO(
        graph,
        particles=20,
        iterations=30,
        beta=0.7,
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


# =============================================================
# REPRODUCIBILITY
# =============================================================

def test_qpso_reproducibility():

    graph1 = build_graph()

    graph2 = build_graph()

    optimizer1 = QPSO(
        graph1,
        particles=20,
        iterations=30,
        beta=0.7,
        seed=42,
    )

    optimizer2 = QPSO(
        graph2,
        particles=20,
        iterations=30,
        beta=0.7,
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
from src.algorithms.adaptive_qpso import AdaptiveQPSO
from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph


def build_test_graph():
    graph = Graph()

    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")

    graph.add_edge(
        Edge(
            a,
            b,
            distance=2,
            travel_time=3,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            b,
            d,
            distance=2,
            travel_time=3,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            a,
            c,
            distance=5,
            travel_time=8,
            congestion=0.5,
        )
    )

    graph.add_edge(
        Edge(
            c,
            d,
            distance=5,
            travel_time=8,
            congestion=0.5,
        )
    )

    return graph


def test_adaptive_qpso():
    graph = build_test_graph()

    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=10,
        iterations=20,
        beta_max=0.9,
        beta_min=0.5,
        seed=42,
        use_dijkstra_seed=False,
    )

    result = optimizer.optimize(
        "A",
        "D",
    )

    assert result["route"] is not None
    assert result["fitness"] < float("inf")

    assert len(result["history"]) == 21

    assert len(
        result["beta_history"]
    ) == 21

    assert (
        result["beta_history"][0]
        >= result["beta_history"][-1]
    )

    print()
    print("Adaptive QPSO")
    print("-----------------------")

    print(
        "Route:",
        [
            node.id
            for node in result["route"]
        ],
    )

    print(
        "Fitness:",
        result["fitness"],
    )

    print(
        "Travel time:",
        result["travel_time"],
    )

    print(
        "Distance:",
        result["distance"],
    )

    print(
        "Congestion:",
        result["congestion"],
    )

    print(
        "Beta start:",
        result["beta_history"][0],
    )

    print(
        "Beta end:",
        result["beta_history"][-1],
    )

    print(
        "Initial fitness:",
        result["history"][0],
    )

    print(
        "Final fitness:",
        result["history"][-1],
    )


if __name__ == "__main__":
    test_adaptive_qpso()
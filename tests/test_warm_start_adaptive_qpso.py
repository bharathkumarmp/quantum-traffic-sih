"""
Warm-start Adaptive-QPSO test.
"""

import time

from src.algorithms.warm_start_adaptive_qpso import (
    WarmStartAdaptiveQPSO,
)

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node


def build_graph():

    graph = Graph()

    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")

    graph.add_edge(
        Edge(
            a,
            b,
            distance=2.0,
            travel_time=2.2,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            b,
            d,
            distance=2.0,
            travel_time=2.2,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            a,
            c,
            distance=3.0,
            travel_time=3.3,
            congestion=0.1,
        )
    )

    graph.add_edge(
        Edge(
            c,
            d,
            distance=3.0,
            travel_time=3.3,
            congestion=0.1,
        )
    )

    return graph


def main():

    graph = build_graph()

    # --------------------------------------------------------
    # Cold start
    # --------------------------------------------------------

    cold_optimizer = (
        WarmStartAdaptiveQPSO(
            graph=graph,
            particles=20,
            iterations=20,
            beta_max=0.9,
            beta_min=0.5,
            seed=42,
            initial_route=None,
        )
    )

    start = time.perf_counter()

    cold_result = (
        cold_optimizer.optimize(
            "A",
            "D",
        )
    )

    cold_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    warm_optimizer = (
        WarmStartAdaptiveQPSO(
            graph=graph,
            particles=20,
            iterations=20,
            beta_max=0.9,
            beta_min=0.5,
            seed=42,
            initial_route=(
                cold_result["route"]
            ),
        )
    )

    start = time.perf_counter()

    warm_result = (
        warm_optimizer.optimize(
            "A",
            "D",
        )
    )

    warm_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "WARM-START ADAPTIVE-QPSO TEST"
    )
    print("=" * 70)

    print()

    print("Cold start:")
    print(
        "  Route   :",
        [
            node.id
            for node in cold_result["route"]
        ],
    )

    print(
        f"  Fitness : "
        f"{cold_result['fitness']:.6f}"
    )

    print(
        f"  Runtime : "
        f"{cold_runtime:.6f} s"
    )

    print()

    print("Warm start:")
    print(
        "  Route   :",
        [
            node.id
            for node in warm_result["route"]
        ],
    )

    print(
        f"  Fitness : "
        f"{warm_result['fitness']:.6f}"
    )

    print(
        f"  Runtime : "
        f"{warm_runtime:.6f} s"
    )

    print(
        f"  Warm started: "
        f"{warm_result['warm_started']}"
    )

    print()

    print(
        "Warm-start test completed."
    )


if __name__ == "__main__":
    main()
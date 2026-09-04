import time

from src.data.graph_generator import LargeGraphGenerator
from src.algorithms.dijkstra import Dijkstra


def test_dijkstra_10000_nodes():

    print()
    print("=" * 70)
    print("10,000 NODE DIJKSTRA SCALABILITY BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Generate large transportation network
    # --------------------------------------------------

    generation_start = time.perf_counter()

    generator = LargeGraphGenerator(
        num_nodes=10_000,
        edges_per_node=3,
        seed=42,
    )

    graph = generator.generate()

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    # --------------------------------------------------
    # 2. Run Dijkstra
    # --------------------------------------------------

    optimizer = Dijkstra(graph)

    start_time = time.perf_counter()

    result = optimizer.optimize(
        "N0",
        "N9999",
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # 3. Validate result
    # --------------------------------------------------

    assert result is not None
    assert result["route"] is not None
    assert len(result["route"]) > 0

    # --------------------------------------------------
    # 4. Display benchmark
    # --------------------------------------------------

    print()
    print("NETWORK")
    print("-" * 70)

    print(
        f"Nodes              : {10_000:,}"
    )

    print(
        f"Edges              : "
        f"{len(graph.get_edges()):,}"
    )

    print(
        f"Generation time    : "
        f"{generation_time:.6f}s"
    )

    print()
    print("ROUTING")
    print("-" * 70)

    print(
        "Source             : N0"
    )

    print(
        "Destination        : N9999"
    )

    print(
        f"Route nodes        : "
        f"{len(result['route'])}"
    )

    print(
        f"Travel time        : "
        f"{result['travel_time']:.4f}"
    )

    print(
        f"Distance           : "
        f"{result['distance']:.4f}"
    )

    print(
        f"Runtime            : "
        f"{runtime:.6f}s"
    )

    print()
    print("=" * 70)
    print("10,000 NODE BENCHMARK COMPLETED")
    print("=" * 70)

    assert runtime > 0
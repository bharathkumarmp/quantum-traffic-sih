import time

from src.data.graph_generator import LargeGraphGenerator
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO


def test_pso_qpso_10000_nodes():

    print()
    print("=" * 70)
    print("10,000 NODE PSO vs QPSO SCALABILITY BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------
    # Generate network
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

    print()
    print("NETWORK")
    print("-" * 70)
    print("Nodes           : 10,000")
    print(
        f"Edges           : "
        f"{len(graph.get_edges()):,}"
    )
    print(
        f"Generation time : "
        f"{generation_time:.6f}s"
    )

    source = "N0"
    destination = "N9999"

    # --------------------------------------------------
    # PSO
    # --------------------------------------------------

    print()
    print("PSO")
    print("-" * 70)

    pso = PSO(
        graph,
        particles=5,
        iterations=5,
        seed=42,
        use_dijkstra_seed=True,
    )

    start = time.perf_counter()

    pso_result = pso.optimize(
        source,
        destination,
    )

    pso_runtime = (
        time.perf_counter()
        - start
    )

    assert pso_result is not None
    assert pso_result["route"] is not None

    print(
        f"Runtime         : "
        f"{pso_runtime:.6f}s"
    )

    print(
        f"Route nodes     : "
        f"{len(pso_result['route'])}"
    )

    print(
        f"Fitness         : "
        f"{pso_result['fitness']:.6f}"
    )

    print(
        f"Travel time     : "
        f"{pso_result['travel_time']:.6f}"
    )

    print(
        f"Distance        : "
        f"{pso_result['distance']:.6f}"
    )

    print(
        f"Congestion      : "
        f"{pso_result['congestion']:.6f}"
    )

    # --------------------------------------------------
    # QPSO
    # --------------------------------------------------

    print()
    print("QPSO")
    print("-" * 70)

    qpso = QPSO(
        graph,
        particles=5,
        iterations=5,
        seed=42,
        use_dijkstra_seed=True,
    )

    start = time.perf_counter()

    qpso_result = qpso.optimize(
        source,
        destination,
    )

    qpso_runtime = (
        time.perf_counter()
        - start
    )

    assert qpso_result is not None
    assert qpso_result["route"] is not None

    print(
        f"Runtime         : "
        f"{qpso_runtime:.6f}s"
    )

    print(
        f"Route nodes     : "
        f"{len(qpso_result['route'])}"
    )

    print(
        f"Fitness         : "
        f"{qpso_result['fitness']:.6f}"
    )

    print(
        f"Travel time     : "
        f"{qpso_result['travel_time']:.6f}"
    )

    print(
        f"Distance        : "
        f"{qpso_result['distance']:.6f}"
    )

    print(
        f"Congestion      : "
        f"{qpso_result['congestion']:.6f}"
    )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"PSO   runtime : "
        f"{pso_runtime:.6f}s"
    )

    print(
        f"QPSO  runtime : "
        f"{qpso_runtime:.6f}s"
    )

    print(
        f"PSO   fitness : "
        f"{pso_result['fitness']:.6f}"
    )

    print(
        f"QPSO  fitness : "
        f"{qpso_result['fitness']:.6f}"
    )

    if pso_result["fitness"] < qpso_result["fitness"]:
        print("Better fitness : PSO")

    elif qpso_result["fitness"] < pso_result["fitness"]:
        print("Better fitness : QPSO")

    else:
        print("Better fitness : Equal")

    print("=" * 70)

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    assert pso_runtime > 0
    assert qpso_runtime > 0
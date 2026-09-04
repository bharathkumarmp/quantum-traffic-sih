import csv
import os
import sys
import time

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.data.graph_generator import LargeGraphGenerator
from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.optimization.fitness import FitnessFunction


# =========================================================
# CONFIGURATION
# =========================================================

NETWORK_SIZES = [
    1000,
    5000,
    10000,
    25000,
    50000,
]

EDGES_PER_NODE = 3

PARTICLES = 20
ITERATIONS = 50

SEED = 42

RESULT_DIR = "results"

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "scalability_benchmark.csv"
)


# =========================================================
# GRAPH GENERATION
# =========================================================

def generate_network(num_nodes):

    print("\n" + "=" * 70)
    print(
        f"GENERATING NETWORK: {num_nodes:,} NODES"
    )
    print("=" * 70)

    start_time = time.perf_counter()

    generator = LargeGraphGenerator(
        num_nodes=num_nodes,
        edges_per_node=EDGES_PER_NODE,
        seed=SEED
    )

    graph = generator.generate()

    generation_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Nodes           : "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges           : "
        f"{len(graph.get_edges()):,}"
    )

    print(
        f"Generation time : "
        f"{generation_time:.6f}s"
    )

    return graph, generation_time


# =========================================================
# DIJKSTRA
# =========================================================

def run_dijkstra(
    graph,
    start,
    destination
):

    print("\nRunning Dijkstra...")

    start_time = time.perf_counter()

    # IMPORTANT:
    # Your Dijkstra class uses optimize(),
    # NOT shortest_path().

    result = Dijkstra(
        graph
    ).optimize(
        start,
        destination
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    route = result["route"]

    if route is None:

        print(
            "Dijkstra could not find a route."
        )

        return {
            "runtime": runtime,
            "route_length": 0,
            "fitness": float("inf"),
            "travel_time": float("inf"),
            "distance": float("inf"),
            "congestion": 1.0,
        }

    print(
        f"Runtime         : "
        f"{runtime:.6f}s"
    )

    print(
        f"Route nodes     : "
        f"{len(route)}"
    )

    print(
        f"Fitness         : "
        f"{result['fitness']:.6f}"
    )

    print(
        f"Travel time     : "
        f"{result['travel_time']:.6f}"
    )

    print(
        f"Distance        : "
        f"{result['distance']:.6f}"
    )

    print(
        f"Congestion      : "
        f"{result['congestion']:.6f}"
    )

    return {
        "runtime": runtime,
        "route_length": len(route),
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
    }


# =========================================================
# PSO
# =========================================================

def run_pso(
    graph,
    start,
    destination
):

    print("\nRunning PSO...")

    fitness_function = FitnessFunction()

    optimizer = PSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=SEED,
        fitness_function=fitness_function,
    )

    start_time = time.perf_counter()

    result = optimizer.optimize(
        start,
        destination
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Runtime         : "
        f"{runtime:.6f}s"
    )

    print(
        f"Route nodes     : "
        f"{len(result['route'])}"
    )

    print(
        f"Fitness         : "
        f"{result['fitness']:.6f}"
    )

    print(
        f"Travel time     : "
        f"{result['travel_time']:.6f}"
    )

    print(
        f"Distance        : "
        f"{result['distance']:.6f}"
    )

    print(
        f"Congestion      : "
        f"{result['congestion']:.6f}"
    )

    return {
        "runtime": runtime,
        "route_length": len(result["route"]),
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
    }


# =========================================================
# QPSO
# =========================================================

def run_qpso(
    graph,
    start,
    destination
):

    print("\nRunning QPSO...")

    fitness_function = FitnessFunction()

    optimizer = QPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=0.7,
        seed=SEED,
        fitness_function=fitness_function,
    )

    start_time = time.perf_counter()

    result = optimizer.optimize(
        start,
        destination
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Runtime         : "
        f"{runtime:.6f}s"
    )

    print(
        f"Route nodes     : "
        f"{len(result['route'])}"
    )

    print(
        f"Fitness         : "
        f"{result['fitness']:.6f}"
    )

    print(
        f"Travel time     : "
        f"{result['travel_time']:.6f}"
    )

    print(
        f"Distance        : "
        f"{result['distance']:.6f}"
    )

    print(
        f"Congestion      : "
        f"{result['congestion']:.6f}"
    )

    return {
        "runtime": runtime,
        "route_length": len(result["route"]),
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
    }


# =========================================================
# MAIN BENCHMARK
# =========================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    all_results = []

    # -----------------------------------------------------
    # TEST EACH NETWORK SIZE
    # -----------------------------------------------------

    for num_nodes in NETWORK_SIZES:

        graph, generation_time = (
            generate_network(num_nodes)
        )

        start = "N0"

        destination = (
            f"N{num_nodes - 1}"
        )

        # =================================================
        # DIJKSTRA
        # =================================================

        dijkstra_result = run_dijkstra(
            graph,
            start,
            destination
        )

        # =================================================
        # PSO
        # =================================================

        pso_result = run_pso(
            graph,
            start,
            destination
        )

        # =================================================
        # QPSO
        # =================================================

        qpso_result = run_qpso(
            graph,
            start,
            destination
        )

        # =================================================
        # SAVE RESULT IN MEMORY
        # =================================================

        result_row = {

            "nodes":
                num_nodes,

            "edges":
                len(graph.get_edges()),

            "generation_time":
                generation_time,

            # -------------------------
            # Dijkstra
            # -------------------------

            "dijkstra_runtime":
                dijkstra_result["runtime"],

            "dijkstra_route_length":
                dijkstra_result[
                    "route_length"
                ],

            "dijkstra_fitness":
                dijkstra_result[
                    "fitness"
                ],

            "dijkstra_travel_time":
                dijkstra_result[
                    "travel_time"
                ],

            "dijkstra_distance":
                dijkstra_result[
                    "distance"
                ],

            "dijkstra_congestion":
                dijkstra_result[
                    "congestion"
                ],

            # -------------------------
            # PSO
            # -------------------------

            "pso_runtime":
                pso_result["runtime"],

            "pso_route_length":
                pso_result[
                    "route_length"
                ],

            "pso_fitness":
                pso_result[
                    "fitness"
                ],

            "pso_travel_time":
                pso_result[
                    "travel_time"
                ],

            "pso_distance":
                pso_result[
                    "distance"
                ],

            "pso_congestion":
                pso_result[
                    "congestion"
                ],

            # -------------------------
            # QPSO
            # -------------------------

            "qpso_runtime":
                qpso_result["runtime"],

            "qpso_route_length":
                qpso_result[
                    "route_length"
                ],

            "qpso_fitness":
                qpso_result[
                    "fitness"
                ],

            "qpso_travel_time":
                qpso_result[
                    "travel_time"
                ],

            "qpso_distance":
                qpso_result[
                    "distance"
                ],

            "qpso_congestion":
                qpso_result[
                    "congestion"
                ],
        }

        all_results.append(
            result_row
        )

    # =====================================================
    # WRITE CSV
    # =====================================================

    if not all_results:
        print(
            "No benchmark results generated."
        )
        return

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n\n")

    print(
        "=" * 110
    )

    print(
        "SCALABILITY BENCHMARK SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Nodes':>8} "
        f"{'Edges':>10} "
        f"{'Dijkstra':>12} "
        f"{'PSO':>12} "
        f"{'QPSO':>12} "
        f"{'PSO Fit':>12} "
        f"{'QPSO Fit':>12}"
    )

    print(
        "-" * 110
    )

    for result in all_results:

        print(
            f"{result['nodes']:>8,} "
            f"{result['edges']:>10,} "
            f"{result['dijkstra_runtime']:>12.6f} "
            f"{result['pso_runtime']:>12.6f} "
            f"{result['qpso_runtime']:>12.6f} "
            f"{result['pso_fitness']:>12.6f} "
            f"{result['qpso_fitness']:>12.6f}"
        )

    print(
        "=" * 110
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        RESULT_FILE
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
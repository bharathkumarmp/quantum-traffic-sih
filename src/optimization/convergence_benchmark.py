"""
Q-ROUTE Convergence Benchmark
=============================

Algorithms compared:

1. PSO
2. QPSO
3. Warm-QPSO
4. Decomposed Warm-QPSO

All algorithms solve the SAME deterministic VRP instance.

Convergence history:

    PSO
        -> optimizer history

    QPSO
        -> optimizer history

    Warm-QPSO
        -> optimizer history

    Decomposed Warm-QPSO
        -> cluster solution histories
        -> merge corresponding solutions
        -> evaluate complete VRP
        -> global VRP fitness history

IMPORTANT
---------
The Decomposed Warm-QPSO convergence curve must NOT be created by
adding cluster fitness values together. The decomposed optimizer
provides a top-level global history, which is used here.

Runtime
-------
Runtime is measured externally around optimizer.optimize() so that
all four algorithms have comparable wall-clock measurements.
"""

import csv
import json
import time

from pathlib import Path

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.algorithms.decomposed_warm_qpso import DecomposedWarmQPSO

from src.data.graph_generator import LargeGraphGenerator

from src.domain.customer import Customer
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem

from src.optimization.vrp_fitness import VRPFitnessFunction


# ================================================================
# BENCHMARK CONFIGURATION
# ================================================================

CUSTOMERS = 100
VEHICLES = 10

PARTICLES = 20
ITERATIONS = 50

CLUSTERS = 10

SEED = 42

EDGES_PER_NODE = 3

WARM_ITERATIONS = 10
WARM_PARTICLES = 10


# ================================================================
# OUTPUT PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

CSV_PATH = (
    RESULTS_DIR / "convergence_benchmark.csv"
)

JSON_PATH = (
    RESULTS_DIR / "convergence_benchmark.json"
)


# ================================================================
# BUILD PROBLEM
# ================================================================

def build_problem():
    """
    Build one deterministic VRP instance.

    The exact same graph, customers, vehicles and fitness function
    are used by every algorithm.
    """

    graph_generator = LargeGraphGenerator(
        num_nodes=CUSTOMERS + 1,
        edges_per_node=EDGES_PER_NODE,
        seed=SEED,
    )

    graph = graph_generator.generate()

    # ------------------------------------------------------------
    # Depot
    # ------------------------------------------------------------

    depot_node = graph.nodes[0]

    depot = str(
        depot_node.id
    )

    # ------------------------------------------------------------
    # Customer locations
    # ------------------------------------------------------------

    available_nodes = [
        node
        for node in graph.nodes
        if str(node.id) != depot
    ]

    customers = []

    for index in range(CUSTOMERS):

        node = available_nodes[
            index % len(available_nodes)
        ]

        customer = Customer(
            id=f"C{index + 1}",
            demand=1.0,
            node_id=str(node.id),
        )

        customers.append(
            customer
        )

    # ------------------------------------------------------------
    # Vehicle capacity
    # ------------------------------------------------------------

    capacity_per_vehicle = max(
        1.0,
        (
            CUSTOMERS / VEHICLES
        ) * 1.25,
    )

    vehicles = []

    for index in range(VEHICLES):

        vehicle = Vehicle(
            id=f"V{index + 1}",
            capacity=capacity_per_vehicle,
            start_node=depot,
            end_node=depot,
        )

        vehicles.append(
            vehicle
        )

    # ------------------------------------------------------------
    # VRP
    # ------------------------------------------------------------

    problem = VRPProblem(
        graph=graph,
        depot=depot,
        customers=customers,
        vehicles=vehicles,
    )

    return problem


# ================================================================
# TIMED OPTIMIZATION
# ================================================================

def timed_optimize(optimizer):
    """
    Run an optimizer and measure wall-clock execution time.

    This is deliberately done outside the optimizer so all four
    algorithms are measured using the same timing method.
    """

    start_time = time.perf_counter()

    result = optimizer.optimize()

    execution_time = (
        time.perf_counter()
        - start_time
    )

    result["execution_time"] = (
        execution_time
    )

    return result


# ================================================================
# EXTRACT GLOBAL DECOMPOSED HISTORY
# ================================================================

def extract_decomposed_history(
    result,
    iterations,
):
    """
    Extract the globally comparable convergence history.

    The current DecomposedWarmQPSO implementation constructs:

        cluster solutions
                |
                v
        merged complete VRP
                |
                v
        global VRP evaluation
                |
                v
        global fitness history

    Therefore we use the top-level `history`.

    We DO NOT sum cluster histories here.

    That old approach would produce:

        cluster_1_fitness
        + cluster_2_fitness
        + ...
        + cluster_N_fitness

    which is not necessarily the fitness of the original VRP.
    """

    history = result.get(
        "history"
    )

    if (
        isinstance(history, list)
        and len(history) > 0
    ):

        return [
            float(value)
            for value in history
        ]

    # ------------------------------------------------------------
    # Fallback.
    #
    # Normally this should not be reached because the current
    # DecomposedWarmQPSO implementation provides a global history.
    # ------------------------------------------------------------

    final_fitness = float(
        result["fitness"]
    )

    return [
        final_fitness
        for _ in range(
            iterations + 1
        )
    ]


# ================================================================
# RUN PSO
# ================================================================

def run_pso(
    problem,
    fitness_function,
):
    """
    Run classical PSO.
    """

    optimizer = VRPPSO(
        problem=problem,
        particles=PARTICLES,
        iterations=ITERATIONS,
        seed=SEED,
        fitness_function=fitness_function,
    )

    result = timed_optimize(
        optimizer
    )

    history = result.get(
        "history",
        [],
    )

    if not history:

        raise RuntimeError(
            "PSO returned an empty "
            "convergence history."
        )

    history = [
        float(value)
        for value in history
    ]

    return result, history


# ================================================================
# RUN QPSO
# ================================================================

def run_qpso(
    problem,
    fitness_function,
):
    """
    Run standard QPSO.
    """

    optimizer = VRPQPSO(
        problem=problem,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=0.7,
        seed=SEED,
        fitness_function=fitness_function,
    )

    result = timed_optimize(
        optimizer
    )

    history = result.get(
        "history",
        [],
    )

    if not history:

        raise RuntimeError(
            "QPSO returned an empty "
            "convergence history."
        )

    history = [
        float(value)
        for value in history
    ]

    return result, history


# ================================================================
# RUN WARM-QPSO
# ================================================================

def run_warm_qpso(
    problem,
    fitness_function,
):
    """
    Run Warm-QPSO.
    """

    optimizer = WarmVRPQPSO(
        problem=problem,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=0.7,
        seed=SEED,
        fitness_function=fitness_function,
        warm_iterations=WARM_ITERATIONS,
        warm_particles=WARM_PARTICLES,
        record_solutions=False,
    )

    result = timed_optimize(
        optimizer
    )

    history = result.get(
        "history",
        [],
    )

    if not history:

        raise RuntimeError(
            "Warm-QPSO returned an empty "
            "convergence history."
        )

    history = [
        float(value)
        for value in history
    ]

    return result, history


# ================================================================
# RUN DECOMPOSED WARM-QPSO
# ================================================================

def run_decomposed_warm_qpso(
    problem,
    fitness_function,
):
    """
    Run Decomposed Warm-QPSO.

    The optimizer internally:

        1. Decomposes the VRP.
        2. Solves cluster-level VRPs.
        3. Merges cluster solution snapshots.
        4. Evaluates the merged complete VRP.
        5. Produces a global convergence history.
    """

    optimizer = DecomposedWarmQPSO(
        problem=problem,
        clusters=CLUSTERS,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=0.7,
        seed=SEED,
        fitness_function=fitness_function,
        warm_iterations=WARM_ITERATIONS,
        warm_particles=WARM_PARTICLES,
    )

    result = timed_optimize(
        optimizer
    )

    history = extract_decomposed_history(
        result,
        ITERATIONS,
    )

    if not history:

        raise RuntimeError(
            "Decomposed Warm-QPSO returned "
            "an empty convergence history."
        )

    return result, history


# ================================================================
# ALGORITHM DISPATCHER
# ================================================================

def run_algorithm(
    algorithm_name,
    problem,
    fitness_function,
):
    """
    Run one of the four algorithms.
    """

    if algorithm_name == "PSO":

        return run_pso(
            problem,
            fitness_function,
        )

    elif algorithm_name == "QPSO":

        return run_qpso(
            problem,
            fitness_function,
        )

    elif algorithm_name == "Warm-QPSO":

        return run_warm_qpso(
            problem,
            fitness_function,
        )

    elif algorithm_name == "Decomposed Warm-QPSO":

        return run_decomposed_warm_qpso(
            problem,
            fitness_function,
        )

    raise ValueError(
        f"Unknown algorithm: {algorithm_name}"
    )


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(records):
    """
    Save convergence data.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "algorithm",
                "iteration",
                "fitness",
            ]
        )

        for record in records:

            writer.writerow(
                [
                    record["algorithm"],
                    record["iteration"],
                    record["fitness"],
                ]
            )

    print(
        f"Saved: {CSV_PATH}"
    )


# ================================================================
# SAVE JSON
# ================================================================

def save_json(
    benchmark_results,
):
    """
    Save benchmark configuration and algorithm results.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "configuration": {
            "customers": CUSTOMERS,
            "vehicles": VEHICLES,
            "particles": PARTICLES,
            "iterations": ITERATIONS,
            "clusters": CLUSTERS,
            "seed": SEED,
            "edges_per_node": EDGES_PER_NODE,
            "warm_iterations": WARM_ITERATIONS,
            "warm_particles": WARM_PARTICLES,
        },

        "algorithms": benchmark_results,
    }

    with open(
        JSON_PATH,
        "w",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"Saved: {JSON_PATH}"
    )


# ================================================================
# PRINT ALGORITHM RESULT
# ================================================================

def print_algorithm_result(
    algorithm_name,
    result,
    history,
):
    """
    Print readable results.
    """

    print()
    print(
        "-" * 70
    )

    print(
        f"Running {algorithm_name}..."
    )

    print(
        "-" * 70
    )

    print(
        f"Final Fitness : "
        f"{float(result['fitness']):.4f}"
    )

    print(
        f"Travel Time   : "
        f"{float(result['travel_time']):.4f}"
    )

    print(
        f"Distance      : "
        f"{float(result['distance']):.4f}"
    )

    print(
        f"Congestion    : "
        f"{float(result['congestion']):.4f}"
    )

    print(
        f"Runtime       : "
        f"{float(result['execution_time']):.4f}s"
    )

    print(
        f"History Size  : "
        f"{len(history)}"
    )


# ================================================================
# MAIN BENCHMARK
# ================================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "Q-ROUTE CONVERGENCE BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        f"Customers  : {CUSTOMERS}"
    )

    print(
        f"Vehicles   : {VEHICLES}"
    )

    print(
        f"Particles  : {PARTICLES}"
    )

    print(
        f"Iterations : {ITERATIONS}"
    )

    print(
        f"Clusters   : {CLUSTERS}"
    )

    print(
        f"Seed       : {SEED}"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------
    # Build one shared VRP.
    # ------------------------------------------------------------

    problem = build_problem()

    # ------------------------------------------------------------
    # Shared global fitness function.
    # ------------------------------------------------------------

    fitness_function = (
        VRPFitnessFunction(
            time_weight=0.5,
            distance_weight=0.3,
            congestion_weight=0.2,
        )
    )

    algorithms = [
        "PSO",
        "QPSO",
        "Warm-QPSO",
        "Decomposed Warm-QPSO",
    ]

    benchmark_results = {}

    csv_records = []

    # ============================================================
    # RUN ALL ALGORITHMS
    # ============================================================

    for algorithm_name in algorithms:

        result, history = run_algorithm(
            algorithm_name,
            problem,
            fitness_function,
        )

        # --------------------------------------------------------
        # Convert history to floats.
        # --------------------------------------------------------

        history = [
            float(value)
            for value in history
        ]

        # --------------------------------------------------------
        # Verify history size.
        # --------------------------------------------------------

        expected_history_size = (
            ITERATIONS + 1
        )

        if len(history) != expected_history_size:

            raise RuntimeError(
                f"{algorithm_name} returned "
                f"{len(history)} history points; "
                f"expected {expected_history_size}."
            )

        # --------------------------------------------------------
        # Verify monotonic best-so-far convergence.
        #
        # Optimization histories should not increase when they
        # represent best-so-far fitness.
        # --------------------------------------------------------

        for index in range(
            1,
            len(history),
        ):

            if (
                history[index]
                >
                history[index - 1]
                + 1e-9
            ):

                raise RuntimeError(
                    f"{algorithm_name} convergence history "
                    "is not monotonic at iteration "
                    f"{index}."
                )

        # --------------------------------------------------------
        # Save CSV rows.
        # --------------------------------------------------------

        for iteration, fitness in enumerate(
            history
        ):

            csv_records.append(
                {
                    "algorithm":
                        algorithm_name,

                    "iteration":
                        iteration,

                    "fitness":
                        fitness,
                }
            )

        # --------------------------------------------------------
        # Save algorithm summary.
        # --------------------------------------------------------

        benchmark_results[
            algorithm_name
        ] = {
            "fitness": float(
                result["fitness"]
            ),

            "travel_time": float(
                result["travel_time"]
            ),

            "distance": float(
                result["distance"]
            ),

            "congestion": float(
                result["congestion"]
            ),

            "execution_time": float(
                result["execution_time"]
            ),

            "history": history,

            "history_size": len(
                history
            ),
        }

        print_algorithm_result(
            algorithm_name,
            result,
            history,
        )

    # ============================================================
    # SAVE FILES
    # ============================================================

    save_csv(
        csv_records
    )

    save_json(
        benchmark_results
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CONVERGENCE BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    for algorithm_name in algorithms:

        history = (
            benchmark_results[
                algorithm_name
            ]["history"]
        )

        runtime = (
            benchmark_results[
                algorithm_name
            ]["execution_time"]
        )

        improvement = (
            (
                history[0]
                - history[-1]
            )
            / history[0]
            * 100.0
        )

        print(
            f"{algorithm_name:<25}"
            f"Initial={history[0]:10.4f} "
            f"Final={history[-1]:10.4f} "
            f"Improvement={improvement:7.2f}% "
            f"Runtime={runtime:.4f}s"
        )

    print(
        "=" * 70
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
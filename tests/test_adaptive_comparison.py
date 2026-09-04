"""
Adaptive QPSO Comparison Experiment

Compares:

1. Classical PSO
2. Fixed QPSO (beta = 0.9)
3. Adaptive QPSO (beta 0.9 -> 0.5)

Across:

    normal
    heavy
    corridor_0
    corridor_1
    incident

10 random seeds per scenario.

Total:
    5 scenarios × 10 seeds × 3 algorithms = 150 runs
"""

import csv
import os
import statistics
import time
import random

from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.algorithms.adaptive_qpso import AdaptiveQPSO

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node


# ============================================================
# CONFIGURATION
# ============================================================

LAYERS = 100
BRANCHES = 3

PARTICLES = 20
ITERATIONS = 50

NETWORK_SEED = 42

SEEDS = list(range(42, 52))

FIXED_BETA = 0.9

ADAPTIVE_BETA_MAX = 0.9
ADAPTIVE_BETA_MIN = 0.5

START_NODE = "N0"
DESTINATION_NODE = f"N{LAYERS}"

RESULT_DIR = "results"

RAW_FILE = os.path.join(
    RESULT_DIR,
    "adaptive_qpso_comparison_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "adaptive_qpso_comparison_summary.csv",
)


# ============================================================
# BUILD NETWORK
# ============================================================

def build_network():
    """
    Build the same controlled layered network used by the
    previous beta-sensitivity experiment.

    Network topology remains fixed across all experiments.
    """

    rng = random.Random(NETWORK_SEED)

    graph = Graph()

    start = Node(START_NODE)

    graph.add_node(start)

    layers = []

    # --------------------------------------------------------
    # Create layers
    # --------------------------------------------------------

    for layer in range(1, LAYERS + 1):

        current_layer = []

        for branch in range(BRANCHES):

            node = Node(
                f"L{layer}_B{branch}"
            )

            graph.add_node(node)

            current_layer.append(node)

        layers.append(current_layer)

    destination = Node(
        DESTINATION_NODE
    )

    graph.add_node(destination)

    # --------------------------------------------------------
    # Start -> first layer
    # --------------------------------------------------------

    for target in layers[0]:

        distance = 5.0
        congestion = 0.10

        travel_time = (
            distance
            * (1.0 + congestion)
        )

        graph.add_edge(
            Edge(
                start,
                target,
                distance=distance,
                travel_time=travel_time,
                congestion=congestion,
            )
        )

    # --------------------------------------------------------
    # Layer -> next layer
    # --------------------------------------------------------

    for layer_index in range(
        LAYERS - 1
    ):

        current_layer = (
            layers[layer_index]
        )

        next_layer = (
            layers[layer_index + 1]
        )

        for source in current_layer:

            for target in next_layer:

                distance = rng.uniform(
                    4.0,
                    6.0,
                )

                congestion = rng.uniform(
                    0.05,
                    0.15,
                )

                travel_time = (
                    distance
                    * (1.0 + congestion)
                )

                graph.add_edge(
                    Edge(
                        source,
                        target,
                        distance=distance,
                        travel_time=travel_time,
                        congestion=congestion,
                    )
                )

    # --------------------------------------------------------
    # Last layer -> destination
    # --------------------------------------------------------

    for source in layers[-1]:

        distance = 5.0
        congestion = 0.10

        travel_time = (
            distance
            * (1.0 + congestion)
        )

        graph.add_edge(
            Edge(
                source,
                destination,
                distance=distance,
                travel_time=travel_time,
                congestion=congestion,
            )
        )

    return graph


# ============================================================
# GET LAYER
# ============================================================

def get_layer_number(node_id):
    """
    Convert:

        L50_B0

    into:

        50

    N0/N100 return None.
    """

    if not node_id.startswith("L"):
        return None

    try:

        value = node_id[1:]

        value = value.split("_")[0]

        return int(value)

    except (
        ValueError,
        IndexError,
    ):
        return None


# ============================================================
# APPLY TRAFFIC SCENARIO
# ============================================================

def apply_scenario(
    graph,
    scenario,
):
    """
    Apply exactly the same traffic scenarios used in the
    beta-sensitivity experiment.
    """

    for edge in graph.get_edges():

        source_id = edge.source.id
        target_id = edge.target.id

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if scenario == "normal":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance
                * 1.10
            )

        # ----------------------------------------------------
        # HEAVY
        # ----------------------------------------------------

        elif scenario == "heavy":

            edge.congestion = 0.50

            edge.travel_time = (
                edge.distance
                * 1.50
            )

        # ----------------------------------------------------
        # CORRIDOR 0
        # ----------------------------------------------------

        elif scenario == "corridor_0":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance
                * 1.10
            )

            source_is_corridor = (
                source_id.startswith("L")
                and "_B0" in source_id
            )

            target_is_corridor = (
                target_id.startswith("L")
                and "_B0" in target_id
            )

            if (
                source_is_corridor
                or target_is_corridor
            ):

                edge.congestion = 0.90

                edge.travel_time = (
                    edge.distance
                    * 5.0
                )

        # ----------------------------------------------------
        # CORRIDOR 1
        # ----------------------------------------------------

        elif scenario == "corridor_1":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance
                * 1.10
            )

            source_is_corridor = (
                source_id.startswith("L")
                and "_B1" in source_id
            )

            target_is_corridor = (
                target_id.startswith("L")
                and "_B1" in target_id
            )

            if (
                source_is_corridor
                or target_is_corridor
            ):

                edge.congestion = 0.90

                edge.travel_time = (
                    edge.distance
                    * 5.0
                )

        # ----------------------------------------------------
        # INCIDENT
        # ----------------------------------------------------

        elif scenario == "incident":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance
                * 1.10
            )

            source_layer = (
                get_layer_number(
                    source_id
                )
            )

            target_layer = (
                get_layer_number(
                    target_id
                )
            )

            if (
                source_layer == 50
                or target_layer == 50
            ):

                edge.congestion = 1.0

                edge.travel_time = (
                    edge.distance
                    * 6.0
                )

        else:

            raise ValueError(
                f"Unknown scenario: "
                f"{scenario}"
            )


# ============================================================
# ROUTE METRICS
# ============================================================

def route_metrics(
    graph,
    route,
):
    """
    Calculate route travel time, distance and
    average congestion.
    """

    if route is None:
        return {
            "travel_time": float("inf"),
            "distance": float("inf"),
            "congestion": float("inf"),
        }

    total_time = 0.0
    total_distance = 0.0

    congestion_values = []

    for source, target in zip(
        route[:-1],
        route[1:],
    ):

        edge = None

        for candidate in (
            graph.get_neighbors(source)
        ):

            if (
                candidate.target.id
                == target.id
                and not candidate.blocked
            ):

                edge = candidate

                break

        if edge is None:

            return {
                "travel_time": float("inf"),
                "distance": float("inf"),
                "congestion": float("inf"),
            }

        total_time += (
            edge.travel_time
        )

        total_distance += (
            edge.distance
        )

        congestion_values.append(
            edge.congestion
        )

    average_congestion = (
        sum(congestion_values)
        / len(congestion_values)
        if congestion_values
        else 0.0
    )

    return {
        "travel_time": total_time,
        "distance": total_distance,
        "congestion": average_congestion,
    }


# ============================================================
# RUN PSO
# ============================================================

def run_pso(
    graph,
    seed,
):
    start_time = time.perf_counter()

    optimizer = PSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        seed=seed,
        use_dijkstra_seed=False,
        initialization_attempts=100,
    )

    result = optimizer.optimize(
        START_NODE,
        DESTINATION_NODE,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


# ============================================================
# RUN FIXED QPSO
# ============================================================

def run_qpso(
    graph,
    seed,
):
    start_time = time.perf_counter()

    optimizer = QPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=FIXED_BETA,
        seed=seed,
        use_dijkstra_seed=False,
        initialization_attempts=100,
    )

    result = optimizer.optimize(
        START_NODE,
        DESTINATION_NODE,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


# ============================================================
# RUN ADAPTIVE QPSO
# ============================================================

def run_adaptive_qpso(
    graph,
    seed,
):
    start_time = time.perf_counter()

    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta_max=ADAPTIVE_BETA_MAX,
        beta_min=ADAPTIVE_BETA_MIN,
        seed=seed,
        use_dijkstra_seed=False,
        initialization_attempts=100,
    )

    result = optimizer.optimize(
        START_NODE,
        DESTINATION_NODE,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


# ============================================================
# RESULT EXTRACTION
# ============================================================

def extract_result(
    algorithm,
    scenario,
    seed,
    result,
    runtime,
):
    history = result.get(
        "history",
        [],
    )

    if history:

        initial_fitness = (
            history[0]
        )

    else:

        initial_fitness = (
            result["fitness"]
        )

    final_fitness = (
        result["fitness"]
    )

    if (
        initial_fitness
        not in (0, float("inf"))
    ):

        improvement = (
            (
                initial_fitness
                - final_fitness
            )
            / initial_fitness
        ) * 100.0

    else:

        improvement = 0.0

    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "seed": seed,
        "fitness": final_fitness,
        "initial_fitness": initial_fitness,
        "improvement_percent": improvement,
        "runtime_seconds": runtime,
        "route_nodes": (
            len(result["route"])
            if result.get("route")
            else 0
        ),
        "travel_time": result.get(
            "travel_time",
            0.0,
        ),
        "distance": result.get(
            "distance",
            0.0,
        ),
        "congestion": result.get(
            "congestion",
            0.0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    scenarios = [
        "normal",
        "heavy",
        "corridor_0",
        "corridor_1",
        "incident",
    ]

    algorithms = [
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
    ]

    raw_results = []

    total_runs = (
        len(scenarios)
        * len(SEEDS)
        * len(algorithms)
    )

    completed = 0

    print()
    print("=" * 100)
    print(
        "PSO vs FIXED QPSO vs ADAPTIVE QPSO"
    )
    print("=" * 100)

    print(
        f"Scenarios       : {len(scenarios)}"
    )

    print(
        f"Seeds           : {len(SEEDS)}"
    )

    print(
        f"Particles       : {PARTICLES}"
    )

    print(
        f"Iterations      : {ITERATIONS}"
    )

    print(
        f"Fixed beta      : {FIXED_BETA}"
    )

    print(
        f"Adaptive beta   : "
        f"{ADAPTIVE_BETA_MAX} -> "
        f"{ADAPTIVE_BETA_MIN}"
    )

    print(
        f"Total runs      : {total_runs}"
    )

    print()

    # ========================================================
    # EXPERIMENT
    # ========================================================

    for scenario in scenarios:

        print()
        print("-" * 100)
        print(
            f"SCENARIO: {scenario}"
        )
        print("-" * 100)

        for seed in SEEDS:

            # ------------------------------------------------
            # IMPORTANT:
            # Each algorithm gets an independent fresh graph.
            # ------------------------------------------------

            graph_pso = build_network()

            apply_scenario(
                graph_pso,
                scenario,
            )

            graph_qpso = build_network()

            apply_scenario(
                graph_qpso,
                scenario,
            )

            graph_adaptive = build_network()

            apply_scenario(
                graph_adaptive,
                scenario,
            )

            # ------------------------------------------------
            # PSO
            # ------------------------------------------------

            result, runtime = run_pso(
                graph_pso,
                seed,
            )

            row = extract_result(
                "PSO",
                scenario,
                seed,
                result,
                runtime,
            )

            raw_results.append(row)

            completed += 1

            print(
                f"[{completed:3d}/{total_runs}] "
                f"{scenario:<12} "
                f"seed={seed} "
                f"PSO={row['fitness']:.6f}"
            )

            # ------------------------------------------------
            # Fixed QPSO
            # ------------------------------------------------

            result, runtime = run_qpso(
                graph_qpso,
                seed,
            )

            row = extract_result(
                "QPSO",
                scenario,
                seed,
                result,
                runtime,
            )

            raw_results.append(row)

            completed += 1

            print(
                f"[{completed:3d}/{total_runs}] "
                f"{scenario:<12} "
                f"seed={seed} "
                f"QPSO={row['fitness']:.6f}"
            )

            # ------------------------------------------------
            # Adaptive QPSO
            # ------------------------------------------------

            result, runtime = (
                run_adaptive_qpso(
                    graph_adaptive,
                    seed,
                )
            )

            row = extract_result(
                "Adaptive-QPSO",
                scenario,
                seed,
                result,
                runtime,
            )

            raw_results.append(row)

            completed += 1

            print(
                f"[{completed:3d}/{total_runs}] "
                f"{scenario:<12} "
                f"seed={seed} "
                f"Adaptive="
                f"{row['fitness']:.6f}"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    fieldnames = list(
        raw_results[0].keys()
    )

    with open(
        RAW_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            raw_results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = []

    for scenario in scenarios:

        for algorithm in algorithms:

            rows = [
                row
                for row in raw_results
                if (
                    row["scenario"]
                    == scenario
                    and row["algorithm"]
                    == algorithm
                )
            ]

            fitness_values = [
                row["fitness"]
                for row in rows
            ]

            runtime_values = [
                row["runtime_seconds"]
                for row in rows
            ]

            improvement_values = [
                row[
                    "improvement_percent"
                ]
                for row in rows
            ]

            congestion_values = [
                row["congestion"]
                for row in rows
            ]

            summary.append(
                {
                    "scenario": scenario,

                    "algorithm": algorithm,

                    "mean_fitness": (
                        statistics.mean(
                            fitness_values
                        )
                    ),

                    "best_fitness": min(
                        fitness_values
                    ),

                    "worst_fitness": max(
                        fitness_values
                    ),

                    "std_fitness": (
                        statistics.stdev(
                            fitness_values
                        )
                        if len(
                            fitness_values
                        ) > 1
                        else 0.0
                    ),

                    "mean_runtime_seconds": (
                        statistics.mean(
                            runtime_values
                        )
                    ),

                    "mean_improvement_percent": (
                        statistics.mean(
                            improvement_values
                        )
                    ),

                    "mean_congestion": (
                        statistics.mean(
                            congestion_values
                        )
                    ),
                }
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_fields = list(
        summary[0].keys()
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 120)
    print(
        "ALGORITHM COMPARISON SUMMARY"
    )
    print("=" * 120)

    print(
        f"{'Scenario':<14}"
        f"{'Algorithm':<18}"
        f"{'Mean Fitness':<16}"
        f"{'Best':<14}"
        f"{'Worst':<14}"
        f"{'Std':<14}"
        f"{'Runtime':<12}"
        f"{'Improvement':<14}"
    )

    print("-" * 120)

    for row in summary:

        print(
            f"{row['scenario']:<14}"
            f"{row['algorithm']:<18}"
            f"{row['mean_fitness']:<16.6f}"
            f"{row['best_fitness']:<14.6f}"
            f"{row['worst_fitness']:<14.6f}"
            f"{row['std_fitness']:<14.6f}"
            f"{row['mean_runtime_seconds']:<12.5f}"
            f"{row['mean_improvement_percent']:<14.2f}%"
        )

    # ========================================================
    # WIN COUNT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "SCENARIO WIN COUNT"
    )
    print("=" * 80)

    win_counts = {
        "PSO": 0,
        "QPSO": 0,
        "Adaptive-QPSO": 0,
    }

    for scenario in scenarios:

        scenario_rows = [
            row
            for row in summary
            if row["scenario"] == scenario
        ]

        best = min(
            scenario_rows,
            key=lambda row:
                row["mean_fitness"],
        )

        win_counts[
            best["algorithm"]
        ] += 1

        print(
            f"{scenario:<14}"
            f"winner = "
            f"{best['algorithm']:<18}"
            f"mean fitness = "
            f"{best['mean_fitness']:.6f}"
        )

    print()
    print(
        f"PSO wins          : "
        f"{win_counts['PSO']}"
    )

    print(
        f"QPSO wins         : "
        f"{win_counts['QPSO']}"
    )

    print(
        f"Adaptive-QPSO wins: "
        f"{win_counts['Adaptive-QPSO']}"
    )

    # ========================================================
    # OVERALL ALGORITHM PERFORMANCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OVERALL ALGORITHM PERFORMANCE"
    )
    print("=" * 80)

    overall = []

    for algorithm in algorithms:

        values = [
            row["mean_fitness"]
            for row in summary
            if row["algorithm"]
            == algorithm
        ]

        runtimes = [
            row[
                "mean_runtime_seconds"
            ]
            for row in summary
            if row["algorithm"]
            == algorithm
        ]

        overall.append(
            {
                "algorithm": algorithm,
                "mean_fitness": (
                    statistics.mean(
                        values
                    )
                ),
                "mean_runtime": (
                    statistics.mean(
                        runtimes
                    )
                ),
            }
        )

    print(
        f"{'Algorithm':<20}"
        f"{'Overall Mean Fitness':<24}"
        f"{'Mean Runtime':<16}"
    )

    print("-" * 60)

    for row in overall:

        print(
            f"{row['algorithm']:<20}"
            f"{row['mean_fitness']:<24.6f}"
            f"{row['mean_runtime']:<16.5f}"
        )

    best_overall = min(
        overall,
        key=lambda row:
            row["mean_fitness"],
    )

    print()
    print(
        "Best overall algorithm: "
        f"{best_overall['algorithm']}"
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print(
        RAW_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "Experiment completed successfully."
    )


if __name__ == "__main__":
    main()
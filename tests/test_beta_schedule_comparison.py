"""
Beta Schedule Comparison Experiment

Compares:

1. Fixed QPSO beta = 0.5
2. Fixed QPSO beta = 0.7
3. Fixed QPSO beta = 0.9
4. Adaptive QPSO beta = 0.9 -> 0.5

Across:
    normal
    heavy
    corridor_0
    corridor_1
    incident

10 seeds per scenario.

Total:
    4 algorithms × 5 scenarios × 10 seeds = 200 runs
"""

import csv
import os
import statistics
import time
import random

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

FIXED_BETAS = [
    0.5,
    0.7,
    0.9,
]

ADAPTIVE_BETA_MAX = 0.9
ADAPTIVE_BETA_MIN = 0.5

START_NODE = "N0"
DESTINATION_NODE = f"N{LAYERS}"

RESULT_DIR = "results"

RAW_FILE = os.path.join(
    RESULT_DIR,
    "beta_schedule_comparison_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "beta_schedule_comparison_summary.csv",
)


# ============================================================
# BUILD CONTROLLED NETWORK
# ============================================================

def build_network():

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
            distance * (1.0 + congestion)
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
            distance * (1.0 + congestion)
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
# GET LAYER NUMBER
# ============================================================

def get_layer_number(node_id):

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

    for edge in graph.get_edges():

        source_id = edge.source.id
        target_id = edge.target.id

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if scenario == "normal":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance * 1.10
            )

        # ----------------------------------------------------
        # HEAVY
        # ----------------------------------------------------

        elif scenario == "heavy":

            edge.congestion = 0.50

            edge.travel_time = (
                edge.distance * 1.50
            )

        # ----------------------------------------------------
        # CORRIDOR 0
        # ----------------------------------------------------

        elif scenario == "corridor_0":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance * 1.10
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
                    edge.distance * 5.0
                )

        # ----------------------------------------------------
        # CORRIDOR 1
        # ----------------------------------------------------

        elif scenario == "corridor_1":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance * 1.10
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
                    edge.distance * 5.0
                )

        # ----------------------------------------------------
        # INCIDENT
        # ----------------------------------------------------

        elif scenario == "incident":

            edge.congestion = 0.10

            edge.travel_time = (
                edge.distance * 1.10
            )

            source_layer = (
                get_layer_number(source_id)
            )

            target_layer = (
                get_layer_number(target_id)
            )

            if (
                source_layer == 50
                or target_layer == 50
            ):

                edge.congestion = 1.0

                edge.travel_time = (
                    edge.distance * 6.0
                )

        else:

            raise ValueError(
                f"Unknown scenario: {scenario}"
            )


# ============================================================
# RUN FIXED QPSO
# ============================================================

def run_fixed_qpso(
    graph,
    seed,
    beta,
):

    start_time = time.perf_counter()

    optimizer = QPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=beta,
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
# EXTRACT RESULT
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

        initial_fitness = history[0]

    else:

        initial_fitness = result["fitness"]

    final_fitness = result["fitness"]

    if (
        initial_fitness != 0
        and initial_fitness != float("inf")
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
# MAIN EXPERIMENT
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
        "QPSO-beta-0.5",
        "QPSO-beta-0.7",
        "QPSO-beta-0.9",
        "Adaptive-QPSO",
    ]

    total_runs = (
        len(scenarios)
        * len(SEEDS)
        * len(algorithms)
    )

    completed = 0

    raw_results = []

    print()
    print("=" * 110)
    print("BETA SCHEDULE COMPARISON")
    print("=" * 110)

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
        f"Fixed betas     : {FIXED_BETAS}"
    )

    print(
        f"Adaptive beta   : "
        f"{ADAPTIVE_BETA_MAX} -> "
        f"{ADAPTIVE_BETA_MIN}"
    )

    print(
        f"Total runs      : {total_runs}"
    )

    # ========================================================
    # SCENARIOS
    # ========================================================

    for scenario in scenarios:

        print()
        print("-" * 110)
        print(
            f"SCENARIO: {scenario}"
        )
        print("-" * 110)

        for seed in SEEDS:

            # ------------------------------------------------
            # Fixed beta 0.5
            # ------------------------------------------------

            graph = build_network()

            apply_scenario(
                graph,
                scenario,
            )

            result, runtime = (
                run_fixed_qpso(
                    graph,
                    seed,
                    0.5,
                )
            )

            row = extract_result(
                "QPSO-beta-0.5",
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
                f"beta=0.5 "
                f"fitness="
                f"{row['fitness']:.6f}"
            )

            # ------------------------------------------------
            # Fixed beta 0.7
            # ------------------------------------------------

            graph = build_network()

            apply_scenario(
                graph,
                scenario,
            )

            result, runtime = (
                run_fixed_qpso(
                    graph,
                    seed,
                    0.7,
                )
            )

            row = extract_result(
                "QPSO-beta-0.7",
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
                f"beta=0.7 "
                f"fitness="
                f"{row['fitness']:.6f}"
            )

            # ------------------------------------------------
            # Fixed beta 0.9
            # ------------------------------------------------

            graph = build_network()

            apply_scenario(
                graph,
                scenario,
            )

            result, runtime = (
                run_fixed_qpso(
                    graph,
                    seed,
                    0.9,
                )
            )

            row = extract_result(
                "QPSO-beta-0.9",
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
                f"beta=0.9 "
                f"fitness="
                f"{row['fitness']:.6f}"
            )

            # ------------------------------------------------
            # Adaptive QPSO
            # ------------------------------------------------

            graph = build_network()

            apply_scenario(
                graph,
                scenario,
            )

            result, runtime = (
                run_adaptive_qpso(
                    graph,
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
                f"adaptive "
                f"fitness="
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
    # CREATE SUMMARY
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
                row["improvement_percent"]
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

        writer.writerows(summary)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 125)
    print("BETA SCHEDULE COMPARISON SUMMARY")
    print("=" * 125)

    print(
        f"{'Scenario':<14}"
        f"{'Algorithm':<20}"
        f"{'Mean Fitness':<16}"
        f"{'Best':<14}"
        f"{'Worst':<14}"
        f"{'Std':<14}"
        f"{'Runtime':<12}"
        f"{'Improvement':<14}"
    )

    print("-" * 125)

    for row in summary:

        print(
            f"{row['scenario']:<14}"
            f"{row['algorithm']:<20}"
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
    print("=" * 90)
    print("BETA STRATEGY WIN COUNT")
    print("=" * 90)

    win_counts = {
        algorithm: 0
        for algorithm in algorithms
    }

    for scenario in scenarios:

        scenario_rows = [
            row
            for row in summary
            if row["scenario"] == scenario
        ]

        winner = min(
            scenario_rows,
            key=lambda row:
                row["mean_fitness"],
        )

        win_counts[
            winner["algorithm"]
        ] += 1

        print(
            f"{scenario:<15}"
            f" winner = "
            f"{winner['algorithm']:<22}"
            f" mean = "
            f"{winner['mean_fitness']:.6f}"
        )

    print()

    for algorithm in algorithms:

        print(
            f"{algorithm:<22}: "
            f"{win_counts[algorithm]}"
        )

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    print()
    print("=" * 90)
    print("OVERALL BETA STRATEGY PERFORMANCE")
    print("=" * 90)

    overall = []

    for algorithm in algorithms:

        values = [
            row["mean_fitness"]
            for row in summary
            if row["algorithm"]
            == algorithm
        ]

        runtimes = [
            row["mean_runtime_seconds"]
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
        f"{'Algorithm':<25}"
        f"{'Overall Mean Fitness':<25}"
        f"{'Mean Runtime':<16}"
    )

    print("-" * 70)

    for row in overall:

        print(
            f"{row['algorithm']:<25}"
            f"{row['mean_fitness']:<25.6f}"
            f"{row['mean_runtime']:<16.5f}"
        )

    overall_winner = min(
        overall,
        key=lambda row:
            row["mean_fitness"],
    )

    print()
    print(
        "Overall best beta strategy: "
        f"{overall_winner['algorithm']}"
    )

    # ========================================================
    # ADAPTIVE VS BEST FIXED
    # ========================================================

    fixed_rows = [
        row
        for row in overall
        if row["algorithm"]
        != "Adaptive-QPSO"
    ]

    best_fixed = min(
        fixed_rows,
        key=lambda row:
            row["mean_fitness"],
    )

    adaptive = next(
        row
        for row in overall
        if row["algorithm"]
        == "Adaptive-QPSO"
    )

    improvement_vs_best_fixed = (
        (
            best_fixed["mean_fitness"]
            - adaptive["mean_fitness"]
        )
        / best_fixed["mean_fitness"]
    ) * 100.0

    print()
    print("=" * 90)
    print("ADAPTIVE VS BEST FIXED BETA")
    print("=" * 90)

    print(
        f"Best fixed strategy : "
        f"{best_fixed['algorithm']}"
    )

    print(
        f"Best fixed fitness  : "
        f"{best_fixed['mean_fitness']:.6f}"
    )

    print(
        f"Adaptive fitness    : "
        f"{adaptive['mean_fitness']:.6f}"
    )

    print(
        f"Adaptive improvement: "
        f"{improvement_vs_best_fixed:.2f}%"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 90)
    print("FILES GENERATED")
    print("=" * 90)

    print(RAW_FILE)
    print(SUMMARY_FILE)

    print()
    print(
        "Experiment completed successfully."
    )


if __name__ == "__main__":
    main()
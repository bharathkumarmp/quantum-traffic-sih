"""
QPSO Beta Sensitivity Experiment

Tests QPSO performance for different beta values across
multiple traffic scenarios and random seeds.

Beta values:
    0.3, 0.5, 0.7, 0.9, 1.0

Scenarios:
    normal
    heavy
    corridor_0
    corridor_1
    incident

Outputs:
    results/qpso_beta_sensitivity_results.csv
    results/qpso_beta_sensitivity_summary.csv
"""

import csv
import os
import random
import statistics
import time

from src.algorithms.qpso import QPSO
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

EXPERIMENT_SEEDS = list(range(42, 52))

BETA_VALUES = [0.3, 0.5, 0.7, 0.9, 1.0]

START_NODE = "N0"
DESTINATION_NODE = f"N{LAYERS}"

RESULT_DIR = "results"

RAW_RESULT_FILE = os.path.join(
    RESULT_DIR,
    "qpso_beta_sensitivity_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "qpso_beta_sensitivity_summary.csv",
)


# ============================================================
# NETWORK GENERATION
# ============================================================

def build_network():
    """
    Build the same controlled layered network used by the
    previous multi-seed traffic benchmark.

    The topology is kept fixed so that beta is the primary
    experimental variable.
    """

    rng = random.Random(NETWORK_SEED)

    graph = Graph()

    # Start node
    start = Node(START_NODE)
    graph.add_node(start)

    # Create layered nodes.
    layers = []

    for layer in range(1, LAYERS + 1):
        current_layer = []

        for branch in range(BRANCHES):
            node_id = f"L{layer}_B{branch}"
            node = Node(node_id)

            graph.add_node(node)
            current_layer.append(node)

        layers.append(current_layer)

    # Destination
    destination = Node(DESTINATION_NODE)
    graph.add_node(destination)

    # --------------------------------------------------------
    # Start -> first layer
    # --------------------------------------------------------

    for target in layers[0]:

        distance = 5.0
        congestion = 0.10
        travel_time = distance * (1.0 + congestion)

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
    # Layer -> layer
    # --------------------------------------------------------

    for layer_index in range(LAYERS - 1):

        current_layer = layers[layer_index]
        next_layer = layers[layer_index + 1]

        for source in current_layer:

            for target in next_layer:

                distance = rng.uniform(4.0, 6.0)
                congestion = rng.uniform(0.05, 0.15)

                travel_time = (
                    distance * (1.0 + congestion)
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
    # Final layer -> destination
    # --------------------------------------------------------

    for source in layers[-1]:

        distance = 5.0
        congestion = 0.10
        travel_time = distance * (1.0 + congestion)

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
# TRAFFIC SCENARIOS
# ============================================================

def apply_scenario(graph, scenario):
    """
    Apply one traffic condition to the network.
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
                edge.distance * 1.10
            )

        # ----------------------------------------------------
        # HEAVY TRAFFIC
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

            if (
                "L" in source_id
                and "_B0" in source_id
            ) or (
                "L" in target_id
                and "_B0" in target_id
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

            if (
                "L" in source_id
                and "_B1" in source_id
            ) or (
                "L" in target_id
                and "_B1" in target_id
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

            source_layer = get_layer_number(source_id)
            target_layer = get_layer_number(target_id)

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


def get_layer_number(node_id):
    """
    Extract layer number from node IDs such as:

        L50_B0

    Returns None for N0 / N100.
    """

    if not node_id.startswith("L"):
        return None

    try:
        remainder = node_id[1:]
        layer_number = remainder.split("_")[0]

        return int(layer_number)

    except (ValueError, IndexError):
        return None


# ============================================================
# RUN QPSO
# ============================================================

def run_qpso(graph, beta, seed):
    """
    Run QPSO for one beta value and one random seed.
    """

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

    runtime = time.perf_counter() - start_time

    return result, runtime


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    os.makedirs(RESULT_DIR, exist_ok=True)

    scenarios = [
        "normal",
        "heavy",
        "corridor_0",
        "corridor_1",
        "incident",
    ]

    raw_results = []

    print()
    print("=" * 80)
    print("QPSO BETA SENSITIVITY EXPERIMENT")
    print("=" * 80)

    print(f"Network layers : {LAYERS}")
    print(f"Branches       : {BRANCHES}")
    print(f"Particles      : {PARTICLES}")
    print(f"Iterations     : {ITERATIONS}")
    print(f"Beta values    : {BETA_VALUES}")
    print(f"Seeds          : {EXPERIMENT_SEEDS}")
    print()

    total_runs = (
        len(scenarios)
        * len(BETA_VALUES)
        * len(EXPERIMENT_SEEDS)
    )

    run_number = 0

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    for scenario in scenarios:

        print()
        print("-" * 80)
        print(f"SCENARIO: {scenario}")
        print("-" * 80)

        for beta in BETA_VALUES:

            fitness_values = []
            runtime_values = []
            convergence_values = []

            for seed in EXPERIMENT_SEEDS:

                run_number += 1

                # Fresh network for every run.
                graph = build_network()

                apply_scenario(
                    graph,
                    scenario,
                )

                result, runtime = run_qpso(
                    graph,
                    beta,
                    seed,
                )

                fitness = result["fitness"]

                history = result.get(
                    "history",
                    [],
                )

                initial_fitness = (
                    history[0]
                    if history
                    else fitness
                )

                final_fitness = fitness

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

                fitness_values.append(fitness)
                runtime_values.append(runtime)
                convergence_values.append(
                    improvement
                )

                raw_results.append(
                    {
                        "scenario": scenario,
                        "beta": beta,
                        "seed": seed,
                        "fitness": fitness,
                        "runtime_seconds": runtime,
                        "initial_fitness": initial_fitness,
                        "final_fitness": final_fitness,
                        "improvement_percent": improvement,
                        "route_nodes": len(
                            result["route"]
                        )
                        if result.get("route")
                        else 0,
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
                )

                print(
                    f"[{run_number:3d}/{total_runs}] "
                    f"{scenario:12s} | "
                    f"beta={beta:.1f} | "
                    f"seed={seed} | "
                    f"fitness={fitness:.6f} | "
                    f"time={runtime:.4f}s"
                )

            print(
                f"  beta={beta:.1f} -> "
                f"mean={statistics.mean(fitness_values):.6f}, "
                f"std={statistics.stdev(fitness_values):.6f}"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    fieldnames = list(raw_results[0].keys())

    with open(
        RAW_RESULT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(raw_results)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = []

    for scenario in scenarios:

        for beta in BETA_VALUES:

            rows = [
                row
                for row in raw_results
                if row["scenario"] == scenario
                and row["beta"] == beta
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
                    "beta": beta,
                    "mean_fitness": statistics.mean(
                        fitness_values
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
                        if len(fitness_values) > 1
                        else 0.0
                    ),
                    "mean_runtime_seconds": statistics.mean(
                        runtime_values
                    ),
                    "mean_improvement_percent": statistics.mean(
                        improvement_values
                    ),
                    "mean_congestion": statistics.mean(
                        congestion_values
                    ),
                }
            )

    summary_fields = list(summary[0].keys())

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
    # PRINT SUMMARY TABLE
    # ========================================================

    print()
    print("=" * 100)
    print("BETA SENSITIVITY SUMMARY")
    print("=" * 100)

    print(
        f"{'Scenario':<14}"
        f"{'Beta':<8}"
        f"{'Mean Fitness':<16}"
        f"{'Best':<14}"
        f"{'Std':<14}"
        f"{'Runtime':<12}"
        f"{'Improvement':<14}"
    )

    print("-" * 100)

    for row in summary:

        print(
            f"{row['scenario']:<14}"
            f"{row['beta']:<8.1f}"
            f"{row['mean_fitness']:<16.6f}"
            f"{row['best_fitness']:<14.6f}"
            f"{row['std_fitness']:<14.6f}"
            f"{row['mean_runtime_seconds']:<12.5f}"
            f"{row['mean_improvement_percent']:<14.2f}%"
        )

    # ========================================================
    # BEST BETA PER SCENARIO
    # ========================================================

    print()
    print("=" * 80)
    print("BEST BETA PER SCENARIO")
    print("=" * 80)

    for scenario in scenarios:

        scenario_rows = [
            row
            for row in summary
            if row["scenario"] == scenario
        ]

        best = min(
            scenario_rows,
            key=lambda row: row["mean_fitness"],
        )

        print(
            f"{scenario:<14} "
            f"best beta = {best['beta']:.1f} "
            f"| mean fitness = "
            f"{best['mean_fitness']:.6f}"
        )

    # ========================================================
    # OVERALL BETA PERFORMANCE
    # ========================================================

    print()
    print("=" * 80)
    print("OVERALL BETA PERFORMANCE")
    print("=" * 80)

    overall = []

    for beta in BETA_VALUES:

        values = [
            row["mean_fitness"]
            for row in summary
            if row["beta"] == beta
        ]

        overall.append(
            (
                beta,
                statistics.mean(values),
                statistics.stdev(values)
                if len(values) > 1
                else 0.0,
            )
        )

    print(
        f"{'Beta':<10}"
        f"{'Overall Mean':<20}"
        f"{'Scenario Std':<15}"
    )

    print("-" * 50)

    for beta, mean_value, std_value in overall:

        print(
            f"{beta:<10.1f}"
            f"{mean_value:<20.6f}"
            f"{std_value:<15.6f}"
        )

    best_overall = min(
        overall,
        key=lambda item: item[1],
    )

    print()
    print(
        f"Best overall beta: "
        f"{best_overall[0]:.1f}"
    )

    print()
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print(RAW_RESULT_FILE)
    print(SUMMARY_FILE)

    print()
    print("Experiment completed.")


if __name__ == "__main__":
    main()
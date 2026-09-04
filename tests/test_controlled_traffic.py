import csv
import os
import random
import sys
import time
import statistics


# =========================================================
# PROJECT PATH
# =========================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO

from src.optimization.fitness import FitnessFunction


# =========================================================
# CONFIGURATION
# =========================================================

LAYERS = 100

BRANCHES = 3

PARTICLES = 20

ITERATIONS = 50

START = "N0"

DESTINATION = f"N{LAYERS}"


# =========================================================
# MULTI-SEED CONFIGURATION
# =========================================================

SEEDS = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
]


# =========================================================
# OUTPUT FILES
# =========================================================

RESULT_DIR = "results"

DETAILED_CSV_FILE = os.path.join(
    RESULT_DIR,
    "controlled_traffic_multiseed_results.csv",
)

SUMMARY_CSV_FILE = os.path.join(
    RESULT_DIR,
    "controlled_traffic_statistical_summary.csv",
)


# =========================================================
# BUILD CONTROLLED LAYERED NETWORK
# =========================================================

def build_network(network_seed=42):
    """
    Build a controlled layered transportation network.

    The network contains:

        START
          |
       Layer 0
          |
       Layer 1
          |
        ...
          |
       Layer 100
          |
      DESTINATION

    Each layer contains BRANCHES parallel nodes.

    Every node in one layer is connected to every
    node in the next layer.
    """

    graph = Graph()

    nodes = {}

    # -----------------------------------------------------
    # Create layered nodes
    # -----------------------------------------------------

    for layer in range(LAYERS + 1):

        for branch in range(BRANCHES):

            node_id = f"N{layer}_{branch}"

            node = Node(node_id)

            nodes[(layer, branch)] = node

            graph.add_node(node)

    # -----------------------------------------------------
    # Start node
    # -----------------------------------------------------

    start_node = Node(START)

    graph.add_node(start_node)

    # -----------------------------------------------------
    # Destination node
    # -----------------------------------------------------

    destination_node = Node(
        DESTINATION
    )

    graph.add_node(destination_node)

    # -----------------------------------------------------
    # Connect start to first layer
    # -----------------------------------------------------

    for branch in range(BRANCHES):

        node = nodes[(0, branch)]

        edge = Edge(
            start_node,
            node,
            distance=5.0,
            travel_time=5.0,
            congestion=0.1,
        )

        graph.add_edge(edge)

    # -----------------------------------------------------
    # Connect every layer
    # -----------------------------------------------------

    rng = random.Random(network_seed)

    for layer in range(LAYERS):

        for source_branch in range(BRANCHES):

            source = nodes[
                (layer, source_branch)
            ]

            for target_branch in range(BRANCHES):

                target = nodes[
                    (
                        layer + 1,
                        target_branch,
                    )
                ]

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

                edge = Edge(
                    source,
                    target,
                    distance=distance,
                    travel_time=travel_time,
                    congestion=congestion,
                )

                graph.add_edge(edge)

    # -----------------------------------------------------
    # Connect final layer to destination
    # -----------------------------------------------------

    for branch in range(BRANCHES):

        source = nodes[
            (LAYERS, branch)
        ]

        edge = Edge(
            source,
            destination_node,
            distance=5.0,
            travel_time=5.0,
            congestion=0.1,
        )

        graph.add_edge(edge)

    return graph


# =========================================================
# NODE INFORMATION HELPERS
# =========================================================

def get_branch(node_id):
    """
    Extract branch number.

    Examples:

        N20_0 -> 0
        N50_1 -> 1
        N100_2 -> 2

    Special nodes such as N0 return None.
    """

    if "_" not in node_id:
        return None

    try:

        return int(
            node_id.rsplit("_", 1)[1]
        )

    except ValueError:

        return None


def get_layer(node_id):
    """
    Extract layer number.

    Examples:

        N20_0 -> 20
        N50_1 -> 50
        N100_2 -> 100
    """

    if "_" not in node_id:
        return None

    try:

        prefix = node_id.split(
            "_",
            1,
        )[0]

        return int(
            prefix[1:]
        )

    except ValueError:

        return None


# =========================================================
# APPLY TRAFFIC SCENARIO
# =========================================================

def apply_traffic_scenario(
    graph,
    scenario,
):
    """
    Apply controlled traffic conditions.

    Scenarios:

        normal
        heavy
        corridor_0
        corridor_1
        incident
    """

    # -----------------------------------------------------
    # Reset all edges
    # -----------------------------------------------------

    for edge in graph.get_edges():

        edge.blocked = False

        edge.congestion = 0.10

        edge.travel_time = (
            edge.distance
            * (1.0 + edge.congestion)
        )

    # =====================================================
    # NORMAL
    # =====================================================

    if scenario == "normal":

        return

    # =====================================================
    # HEAVY TRAFFIC
    # =====================================================

    if scenario == "heavy":

        for edge in graph.get_edges():

            edge.congestion = 0.50

            edge.travel_time = (
                edge.distance
                * (1.0 + edge.congestion)
            )

        return

    # =====================================================
    # CORRIDOR 0
    # =====================================================

    if scenario == "corridor_0":

        for edge in graph.get_edges():

            source_branch = get_branch(
                edge.source.id
            )

            target_branch = get_branch(
                edge.target.id
            )

            if (
                source_branch == 0
                or target_branch == 0
            ):

                edge.congestion = 0.90

                edge.travel_time = (
                    edge.distance
                    * 5.0
                )

        return

    # =====================================================
    # CORRIDOR 1
    # =====================================================

    if scenario == "corridor_1":

        for edge in graph.get_edges():

            source_branch = get_branch(
                edge.source.id
            )

            target_branch = get_branch(
                edge.target.id
            )

            if (
                source_branch == 1
                or target_branch == 1
            ):

                edge.congestion = 0.90

                edge.travel_time = (
                    edge.distance
                    * 5.0
                )

        return

    # =====================================================
    # INCIDENT AT LAYER 50
    # =====================================================

    if scenario == "incident":

        for edge in graph.get_edges():

            source_layer = get_layer(
                edge.source.id
            )

            target_layer = get_layer(
                edge.target.id
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

        return

    # =====================================================
    # UNKNOWN SCENARIO
    # =====================================================

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


# =========================================================
# ROUTE METRICS
# =========================================================

def route_metrics(
    graph,
    route,
):

    if route is None or len(route) < 2:

        return (
            float("inf"),
            float("inf"),
            1.0,
        )

    total_time = 0.0

    total_distance = 0.0

    congestion_values = []

    for source, target in zip(
        route[:-1],
        route[1:],
    ):

        edge_found = False

        for edge in graph.get_neighbors(
            source
        ):

            if (
                edge.target.id == target.id
                and not edge.blocked
            ):

                total_time += (
                    edge.travel_time
                )

                total_distance += (
                    edge.distance
                )

                congestion_values.append(
                    edge.congestion
                )

                edge_found = True

                break

        if not edge_found:

            return (
                float("inf"),
                float("inf"),
                1.0,
            )

    if congestion_values:

        average_congestion = (
            sum(congestion_values)
            / len(congestion_values)
        )

    else:

        average_congestion = 0.0

    return (
        total_time,
        total_distance,
        average_congestion,
    )


# =========================================================
# RUN PSO
# =========================================================

def run_pso(
    graph,
    fitness_function,
    seed,
):

    optimizer = PSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=seed,
        fitness_function=fitness_function,

        # Research mode
        use_dijkstra_seed=False,

        initialization_attempts=100,
    )

    start_time = time.perf_counter()

    result = optimizer.optimize(
        START,
        DESTINATION,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


# =========================================================
# RUN QPSO
# =========================================================

def run_qpso(
    graph,
    fitness_function,
    seed,
):

    optimizer = QPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=0.7,
        seed=seed,
        fitness_function=fitness_function,

        # Research mode
        use_dijkstra_seed=False,

        initialization_attempts=100,
    )

    start_time = time.perf_counter()

    result = optimizer.optimize(
        START,
        DESTINATION,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


# =========================================================
# SAFE MEAN
# =========================================================

def safe_mean(values):

    valid_values = [
        value
        for value in values
        if value != float("inf")
    ]

    if not valid_values:
        return float("inf")

    return statistics.mean(
        valid_values
    )


# =========================================================
# SAFE STANDARD DEVIATION
# =========================================================

def safe_std(values):

    valid_values = [
        value
        for value in values
        if value != float("inf")
    ]

    if len(valid_values) < 2:
        return 0.0

    return statistics.stdev(
        valid_values
    )


# =========================================================
# CALCULATE IMPROVEMENT
# =========================================================

def calculate_improvement(
    pso_fitness,
    qpso_fitness,
):
    """
    Positive value means QPSO is better.

        improvement =
        (PSO - QPSO) / PSO * 100
    """

    if (
        pso_fitness == float("inf")
        or qpso_fitness == float("inf")
    ):
        return 0.0

    if pso_fitness == 0:
        return 0.0

    return (
        (
            pso_fitness
            - qpso_fitness
        )
        / abs(pso_fitness)
    ) * 100.0


# =========================================================
# MAIN
# =========================================================

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

    detailed_results = []

    summary_results = []

    total_runs = (
        len(scenarios)
        * len(SEEDS)
    )

    completed_runs = 0

    print()
    print("=" * 100)
    print("MULTI-SEED CONTROLLED TRAFFIC BENCHMARK")
    print("=" * 100)

    print()
    print(
        f"Layers       : {LAYERS}"
    )

    print(
        f"Branches     : {BRANCHES}"
    )

    print(
        f"Particles    : {PARTICLES}"
    )

    print(
        f"Iterations   : {ITERATIONS}"
    )

    print(
        f"Seeds        : {SEEDS}"
    )

    print(
        f"Scenarios    : {len(scenarios)}"
    )

    print(
        f"Total runs   : {total_runs * 2}"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # RUN ALL SCENARIOS
    # =====================================================

    for scenario in scenarios:

        print()
        print("=" * 100)
        print(
            f"SCENARIO: {scenario.upper()}"
        )
        print("=" * 100)

        scenario_rows = []

        # -------------------------------------------------
        # RUN EACH SEED
        # -------------------------------------------------

        for seed in SEEDS:

            print()
            print(
                f"Seed {seed}:"
            )

            # ---------------------------------------------
            # Build fresh network
            # ---------------------------------------------

            graph = build_network(
                network_seed=42
            )

            # ---------------------------------------------
            # Apply traffic scenario
            # ---------------------------------------------

            apply_traffic_scenario(
                graph,
                scenario,
            )

            # ---------------------------------------------
            # Fitness
            # ---------------------------------------------

            fitness_function = FitnessFunction(
                time_weight=0.5,
                distance_weight=0.3,
                congestion_weight=0.2,
            )

            # =============================================
            # PSO
            # =============================================

            try:

                pso_result, pso_runtime = run_pso(
                    graph,
                    fitness_function,
                    seed,
                )

            except Exception as error:

                print(
                    f"  PSO ERROR: {error}"
                )

                pso_result = {
                    "route": None,
                    "fitness": float("inf"),
                    "travel_time": float("inf"),
                    "distance": float("inf"),
                    "congestion": 1.0,
                    "history": [],
                }

                pso_runtime = float("inf")

            # =============================================
            # QPSO
            # =============================================

            try:

                qpso_result, qpso_runtime = run_qpso(
                    graph,
                    fitness_function,
                    seed,
                )

            except Exception as error:

                print(
                    f"  QPSO ERROR: {error}"
                )

                qpso_result = {
                    "route": None,
                    "fitness": float("inf"),
                    "travel_time": float("inf"),
                    "distance": float("inf"),
                    "congestion": 1.0,
                    "history": [],
                }

                qpso_runtime = float("inf")

            # ---------------------------------------------
            # Extract values
            # ---------------------------------------------

            pso_fitness = (
                pso_result["fitness"]
            )

            qpso_fitness = (
                qpso_result["fitness"]
            )

            pso_congestion = (
                pso_result["congestion"]
            )

            qpso_congestion = (
                qpso_result["congestion"]
            )

            pso_travel_time = (
                pso_result["travel_time"]
            )

            qpso_travel_time = (
                qpso_result["travel_time"]
            )

            pso_distance = (
                pso_result["distance"]
            )

            qpso_distance = (
                qpso_result["distance"]
            )

            pso_route_length = (
                len(pso_result["route"])
                if pso_result["route"]
                else 0
            )

            qpso_route_length = (
                len(qpso_result["route"])
                if qpso_result["route"]
                else 0
            )

            # ---------------------------------------------
            # Convergence
            # ---------------------------------------------

            pso_history = (
                pso_result.get(
                    "history",
                    [],
                )
            )

            qpso_history = (
                qpso_result.get(
                    "history",
                    [],
                )
            )

            if pso_history:

                pso_initial = pso_history[0]
                pso_final = pso_history[-1]

            else:

                pso_initial = pso_fitness
                pso_final = pso_fitness

            if qpso_history:

                qpso_initial = qpso_history[0]
                qpso_final = qpso_history[-1]

            else:

                qpso_initial = qpso_fitness
                qpso_final = qpso_fitness

            # ---------------------------------------------
            # Improvement during optimization
            # ---------------------------------------------

            if pso_initial != 0:

                pso_improvement = (
                    (
                        pso_initial
                        - pso_final
                    )
                    / abs(pso_initial)
                ) * 100.0

            else:

                pso_improvement = 0.0

            if qpso_initial != 0:

                qpso_improvement = (
                    (
                        qpso_initial
                        - qpso_final
                    )
                    / abs(qpso_initial)
                ) * 100.0

            else:

                qpso_improvement = 0.0

            # ---------------------------------------------
            # QPSO versus PSO
            # ---------------------------------------------

            comparison_improvement = (
                calculate_improvement(
                    pso_fitness,
                    qpso_fitness,
                )
            )

            if (
                qpso_fitness
                < pso_fitness
            ):

                winner = "QPSO"

            elif (
                pso_fitness
                < qpso_fitness
            ):

                winner = "PSO"

            else:

                winner = "Equal"

            # ---------------------------------------------
            # Print run
            # ---------------------------------------------

            print(
                f"  PSO  fitness : "
                f"{pso_fitness:.6f}"
            )

            print(
                f"  QPSO fitness : "
                f"{qpso_fitness:.6f}"
            )

            print(
                f"  Winner       : "
                f"{winner}"
            )

            print(
                f"  QPSO vs PSO  : "
                f"{comparison_improvement:.2f}%"
            )

            # ---------------------------------------------
            # Detailed row
            # ---------------------------------------------

            row = {
                "scenario": scenario,
                "seed": seed,

                "pso_runtime": pso_runtime,
                "qpso_runtime": qpso_runtime,

                "pso_route_length": pso_route_length,
                "qpso_route_length": qpso_route_length,

                "pso_fitness": pso_fitness,
                "qpso_fitness": qpso_fitness,

                "pso_travel_time": pso_travel_time,
                "qpso_travel_time": qpso_travel_time,

                "pso_distance": pso_distance,
                "qpso_distance": qpso_distance,

                "pso_congestion": pso_congestion,
                "qpso_congestion": qpso_congestion,

                "pso_initial_fitness": pso_initial,
                "pso_final_fitness": pso_final,

                "qpso_initial_fitness": qpso_initial,
                "qpso_final_fitness": qpso_final,

                "pso_improvement_percent":
                    pso_improvement,

                "qpso_improvement_percent":
                    qpso_improvement,

                "qpso_vs_pso_improvement_percent":
                    comparison_improvement,

                "winner": winner,
            }

            detailed_results.append(row)

            scenario_rows.append(row)

            completed_runs += 2

            print(
                f"  Progress     : "
                f"{completed_runs}/"
                f"{total_runs * 2}"
            )

        # =================================================
        # STATISTICAL SUMMARY
        # =================================================

        pso_fitness_values = [
            row["pso_fitness"]
            for row in scenario_rows
        ]

        qpso_fitness_values = [
            row["qpso_fitness"]
            for row in scenario_rows
        ]

        pso_runtime_values = [
            row["pso_runtime"]
            for row in scenario_rows
        ]

        qpso_runtime_values = [
            row["qpso_runtime"]
            for row in scenario_rows
        ]

        pso_congestion_values = [
            row["pso_congestion"]
            for row in scenario_rows
        ]

        qpso_congestion_values = [
            row["qpso_congestion"]
            for row in scenario_rows
        ]

        # ---------------------------------------------
        # Winners
        # ---------------------------------------------

        qpso_wins = sum(
            1
            for row in scenario_rows
            if row["winner"] == "QPSO"
        )

        pso_wins = sum(
            1
            for row in scenario_rows
            if row["winner"] == "PSO"
        )

        equal_results = sum(
            1
            for row in scenario_rows
            if row["winner"] == "Equal"
        )

        # ---------------------------------------------
        # Mean QPSO improvement
        # ---------------------------------------------

        improvement_values = [
            row[
                "qpso_vs_pso_improvement_percent"
            ]
            for row in scenario_rows
        ]

        # ---------------------------------------------
        # Summary
        # ---------------------------------------------

        summary = {

            "scenario": scenario,

            "runs": len(scenario_rows),

            # PSO
            "pso_mean_fitness":
                safe_mean(
                    pso_fitness_values
                ),

            "pso_best_fitness":
                min(
                    pso_fitness_values
                ),

            "pso_worst_fitness":
                max(
                    pso_fitness_values
                ),

            "pso_std_fitness":
                safe_std(
                    pso_fitness_values
                ),

            "pso_mean_runtime":
                safe_mean(
                    pso_runtime_values
                ),

            "pso_mean_congestion":
                safe_mean(
                    pso_congestion_values
                ),

            # QPSO
            "qpso_mean_fitness":
                safe_mean(
                    qpso_fitness_values
                ),

            "qpso_best_fitness":
                min(
                    qpso_fitness_values
                ),

            "qpso_worst_fitness":
                max(
                    qpso_fitness_values
                ),

            "qpso_std_fitness":
                safe_std(
                    qpso_fitness_values
                ),

            "qpso_mean_runtime":
                safe_mean(
                    qpso_runtime_values
                ),

            "qpso_mean_congestion":
                safe_mean(
                    qpso_congestion_values
                ),

            # Comparison
            "mean_qpso_vs_pso_improvement_percent":
                safe_mean(
                    improvement_values
                ),

            "pso_wins":
                pso_wins,

            "qpso_wins":
                qpso_wins,

            "equal":
                equal_results,
        }

        summary_results.append(
            summary
        )

        # ---------------------------------------------
        # Print scenario statistics
        # ---------------------------------------------

        print()
        print(
            "-" * 100
        )

        print(
            f"{scenario.upper()} STATISTICS"
        )

        print(
            "-" * 100
        )

        print(
            f"PSO  mean fitness : "
            f"{summary['pso_mean_fitness']:.6f}"
        )

        print(
            f"PSO  best fitness : "
            f"{summary['pso_best_fitness']:.6f}"
        )

        print(
            f"PSO  worst fitness: "
            f"{summary['pso_worst_fitness']:.6f}"
        )

        print(
            f"PSO  std dev      : "
            f"{summary['pso_std_fitness']:.6f}"
        )

        print(
            f"PSO  mean runtime : "
            f"{summary['pso_mean_runtime']:.6f}s"
        )

        print()

        print(
            f"QPSO mean fitness : "
            f"{summary['qpso_mean_fitness']:.6f}"
        )

        print(
            f"QPSO best fitness : "
            f"{summary['qpso_best_fitness']:.6f}"
        )

        print(
            f"QPSO worst fitness: "
            f"{summary['qpso_worst_fitness']:.6f}"
        )

        print(
            f"QPSO std dev      : "
            f"{summary['qpso_std_fitness']:.6f}"
        )

        print(
            f"QPSO mean runtime : "
            f"{summary['qpso_mean_runtime']:.6f}s"
        )

        print()

        print(
            f"Mean QPSO vs PSO  : "
            f"{summary['mean_qpso_vs_pso_improvement_percent']:.2f}%"
        )

        print(
            f"PSO wins          : "
            f"{pso_wins}"
        )

        print(
            f"QPSO wins         : "
            f"{qpso_wins}"
        )

        print(
            f"Equal             : "
            f"{equal_results}"
        )

        print(
            "-" * 100
        )

    # =====================================================
    # SAVE DETAILED CSV
    # =====================================================

    if detailed_results:

        with open(
            DETAILED_CSV_FILE,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    detailed_results[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                detailed_results
            )

    # =====================================================
    # SAVE SUMMARY CSV
    # =====================================================

    if summary_results:

        with open(
            SUMMARY_CSV_FILE,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    summary_results[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                summary_results
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print()
    print("=" * 130)

    print(
        "MULTI-SEED CONTROLLED TRAFFIC "
        "STATISTICAL SUMMARY"
    )

    print("=" * 130)

    print()

    print(
        f"{'Scenario':<15}"
        f"{'PSO Mean':>13}"
        f"{'QPSO Mean':>13}"
        f"{'PSO Std':>12}"
        f"{'QPSO Std':>12}"
        f"{'QPSO Improve':>15}"
        f"{'PSO Wins':>10}"
        f"{'QPSO Wins':>11}"
    )

    print(
        "-" * 130
    )

    for row in summary_results:

        print(
            f"{row['scenario']:<15}"
            f"{row['pso_mean_fitness']:>13.6f}"
            f"{row['qpso_mean_fitness']:>13.6f}"
            f"{row['pso_std_fitness']:>12.6f}"
            f"{row['qpso_std_fitness']:>12.6f}"
            f"{row['mean_qpso_vs_pso_improvement_percent']:>14.2f}%"
            f"{row['pso_wins']:>10}"
            f"{row['qpso_wins']:>11}"
        )

    print(
        "=" * 130
    )

    # =====================================================
    # OVERALL WIN COUNT
    # =====================================================

    total_pso_wins = sum(
        row["pso_wins"]
        for row in summary_results
    )

    total_qpso_wins = sum(
        row["qpso_wins"]
        for row in summary_results
    )

    total_equal = sum(
        row["equal"]
        for row in summary_results
    )

    total_comparisons = (
        total_pso_wins
        + total_qpso_wins
        + total_equal
    )

    print()

    print(
        "OVERALL RESULTS"
    )

    print(
        "-" * 60
    )

    print(
        f"Total comparisons : "
        f"{total_comparisons}"
    )

    print(
        f"PSO wins          : "
        f"{total_pso_wins}"
    )

    print(
        f"QPSO wins         : "
        f"{total_qpso_wins}"
    )

    print(
        f"Equal             : "
        f"{total_equal}"
    )

    if total_comparisons > 0:

        print(
            f"PSO win rate      : "
            f"{(total_pso_wins / total_comparisons) * 100:.2f}%"
        )

        print(
            f"QPSO win rate     : "
            f"{(total_qpso_wins / total_comparisons) * 100:.2f}%"
        )

    print()

    print(
        "Detailed results saved to:"
    )

    print(
        DETAILED_CSV_FILE
    )

    print()

    print(
        "Statistical summary saved to:"
    )

    print(
        SUMMARY_CSV_FILE
    )

    print()

    print(
        "=" * 130
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
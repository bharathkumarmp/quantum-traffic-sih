"""
SIH Dynamic Disruption Benchmark

Purpose
-------
Evaluate route optimization under localized and moving
traffic disruptions.

Algorithms
---------
1. Dijkstra
2. PSO
3. QPSO
4. Adaptive-QPSO
5. Warm-Start Adaptive-QPSO

Traffic scenarios
-----------------
1. normal
2. corridor_0
3. corridor_1
4. moving_corridor
5. incident_50
6. recovery

The experiment is repeated across multiple network seeds.
"""

import csv
import os
import random
import time
import statistics

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.algorithms.adaptive_qpso import AdaptiveQPSO
from src.algorithms.warm_start_adaptive_qpso import (
    WarmStartAdaptiveQPSO,
)

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node

from src.optimization.fitness import FitnessFunction
from src.traffic.traffic_model import DynamicTrafficModel


# ============================================================
# CONFIGURATION
# ============================================================

LAYERS = 100
BRANCHES = 3

PARTICLES = 20

PSO_ITERATIONS = 50
QPSO_ITERATIONS = 50
ADAPTIVE_ITERATIONS = 50
WARM_ITERATIONS = 30

PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.5
PSO_SOCIAL = 1.5

QPSO_BETA = 0.9

BETA_MAX = 0.9
BETA_MIN = 0.5

NETWORK_SEEDS = list(range(42, 52))

ALGORITHM_SEED = 100

START = "N0"
DESTINATION = f"N{LAYERS + 1}"

RESULTS_DIR = "results"

RAW_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "disruption_benchmark_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "disruption_benchmark_summary.csv",
)

STATE_SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "disruption_state_summary.csv",
)


# ============================================================
# FITNESS
# ============================================================

fitness_function = FitnessFunction(
    time_weight=0.5,
    distance_weight=0.3,
    congestion_weight=0.2,
)


# ============================================================
# GRAPH GENERATOR
# ============================================================

def build_layered_graph(
    layers=LAYERS,
    branches=BRANCHES,
    seed=42,
):
    """
    Create a layered transportation network.

    Each layer contains exactly three branches.

    Node numbering:

        N1 N2 N3       -> Layer 1
        N4 N5 N6       -> Layer 2
        ...

    The graph contains a start node N0 and
    destination node N101.
    """

    graph = Graph()

    rng = random.Random(seed)

    start_node = Node("N0")

    graph.add_node(start_node)

    layer_nodes = {}

    node_counter = 1

    # --------------------------------------------------------
    # Create layers
    # --------------------------------------------------------

    for layer in range(
        1,
        layers + 1,
    ):

        current_layer = []

        for branch in range(
            branches
        ):

            node = Node(
                f"N{node_counter}"
            )

            graph.add_node(node)

            current_layer.append(node)

            node_counter += 1

        layer_nodes[layer] = current_layer

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    destination_node = Node(
        f"N{layers + 1}"
    )

    graph.add_node(
        destination_node
    )

    # --------------------------------------------------------
    # Start -> Layer 1
    # --------------------------------------------------------

    for branch, node in enumerate(
        layer_nodes[1]
    ):

        # Slight branch-specific difference
        # creates meaningful route choices.
        distance = (
            5.0
            + branch * 0.15
        )

        congestion = 0.10

        travel_time = (
            distance
            * (
                1.0
                + congestion
            )
        )

        graph.add_edge(
            Edge(
                start_node,
                node,
                distance=distance,
                travel_time=travel_time,
                congestion=congestion,
            )
        )

    # --------------------------------------------------------
    # Layer -> next layer
    # --------------------------------------------------------

    for layer in range(
        1,
        layers,
    ):

        current_layer = layer_nodes[
            layer
        ]

        next_layer = layer_nodes[
            layer + 1
        ]

        for source_branch, source in enumerate(
            current_layer
        ):

            for target_branch, target in enumerate(
                next_layer
            ):

                distance = (
                    rng.uniform(
                        4.0,
                        6.0,
                    )
                    + source_branch * 0.03
                    + target_branch * 0.03
                )

                congestion = (
                    rng.uniform(
                        0.05,
                        0.15,
                    )
                )

                travel_time = (
                    distance
                    * (
                        1.0
                        + congestion
                    )
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

    for branch, node in enumerate(
        layer_nodes[layers]
    ):

        distance = (
            5.0
            + branch * 0.15
        )

        congestion = 0.10

        travel_time = (
            distance
            * (
                1.0
                + congestion
            )
        )

        graph.add_edge(
            Edge(
                node,
                destination_node,
                distance=distance,
                travel_time=travel_time,
                congestion=congestion,
            )
        )

    return graph


# ============================================================
# NODE -> LAYER
# ============================================================

def node_layer(
    node_id,
):
    """
    Convert node ID to actual layer.

    N1,N2,N3 -> layer 1
    N4,N5,N6 -> layer 2
    """

    number = int(
        str(node_id).replace(
            "N",
            "",
        )
    )

    if number == 0:
        return 0

    return (
        (number - 1)
        // BRANCHES
    ) + 1


# ============================================================
# NODE -> BRANCH
# ============================================================

def node_branch(
    node_id,
):
    """
    Return branch number.

    Branches:

        0
        1
        2
    """

    number = int(
        str(node_id).replace(
            "N",
            "",
        )
    )

    if number == 0:
        return -1

    return (
        number - 1
    ) % BRANCHES


# ============================================================
# EDGE UPDATE
# ============================================================

def update_edge(
    edge,
    congestion,
):
    """
    Update an edge using the same traffic model
    used throughout the project.
    """

    congestion = max(
        0.0,
        min(
            1.0,
            float(congestion),
        ),
    )

    edge.congestion = congestion

    edge.travel_time = (
        edge.distance
        * (
            1.0
            + 5.0 * congestion
        )
    )


# ============================================================
# RESET TRAFFIC
# ============================================================

def reset_traffic(
    graph,
):
    """
    Reset all edges to normal traffic.
    """

    for edge in graph.get_edges():

        update_edge(
            edge,
            0.10,
        )


# ============================================================
# CORRIDOR CONGESTION
# ============================================================

def apply_corridor(
    graph,
    branch,
    congestion=0.90,
):
    """
    Apply congestion to an entire branch corridor.

    An edge belongs to a corridor if either endpoint
    belongs to the selected branch.

    Other edges remain at normal congestion.
    """

    for edge in graph.get_edges():

        source_branch = node_branch(
            edge.source.id
        )

        target_branch = node_branch(
            edge.target.id
        )

        if (
            source_branch == branch
            or target_branch == branch
        ):

            update_edge(
                edge,
                congestion,
            )

        else:

            update_edge(
                edge,
                0.10,
            )


# ============================================================
# INCIDENT
# ============================================================

def apply_incident(
    graph,
    layer,
    congestion=1.0,
):
    """
    Apply severe congestion around a specific layer.
    """

    for edge in graph.get_edges():

        source_layer = node_layer(
            edge.source.id
        )

        target_layer = node_layer(
            edge.target.id
        )

        if (
            source_layer == layer
            or target_layer == layer
        ):

            update_edge(
                edge,
                congestion,
            )

        else:

            update_edge(
                edge,
                0.10,
            )


# ============================================================
# MOVING CORRIDOR
# ============================================================

def apply_moving_corridor(
    graph,
    branch,
):
    """
    Move congestion to a different corridor.
    """

    apply_corridor(
        graph,
        branch=branch,
        congestion=0.90,
    )


# ============================================================
# TRAFFIC STATE
# ============================================================

def apply_state(
    graph,
    state,
):
    """
    Apply one benchmark state.
    """

    reset_traffic(
        graph
    )

    if state == "normal":

        return

    if state == "corridor_0":

        apply_corridor(
            graph,
            branch=0,
            congestion=0.90,
        )

        return

    if state == "corridor_1":

        apply_corridor(
            graph,
            branch=1,
            congestion=0.90,
        )

        return

    if state == "moving_corridor":

        apply_moving_corridor(
            graph,
            branch=2,
        )

        return

    if state == "incident_50":

        apply_incident(
            graph,
            layer=50,
            congestion=1.0,
        )

        return

    if state == "recovery":

        reset_traffic(
            graph
        )

        return

    raise ValueError(
        f"Unknown traffic state: {state}"
    )


# ============================================================
# ROUTE METRICS
# ============================================================

def route_metrics(
    graph,
    route,
):
    """
    Calculate route-level metrics.
    """

    if (
        route is None
        or len(route) < 2
    ):
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

        selected_edge = None

        for edge in graph.get_neighbors(
            source
        ):

            if (
                edge.target.id
                == target.id
                and not edge.blocked
            ):

                selected_edge = edge
                break

        if selected_edge is None:

            return (
                float("inf"),
                float("inf"),
                1.0,
            )

        total_time += (
            selected_edge.travel_time
        )

        total_distance += (
            selected_edge.distance
        )

        congestion_values.append(
            selected_edge.congestion
        )

    average_congestion = (
        sum(congestion_values)
        / len(congestion_values)
    )

    return (
        total_time,
        total_distance,
        average_congestion,
    )


# ============================================================
# ROUTE SIGNATURE
# ============================================================

def route_signature(
    route,
):
    if route is None:
        return None

    return tuple(
        str(node.id)
        for node in route
    )


# ============================================================
# DIJKSTRA
# ============================================================

def run_dijkstra(
    graph,
):
    optimizer = Dijkstra(
        graph=graph,
        fitness_function=fitness_function,
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

    route = result["route"]

    (
        travel_time,
        distance,
        congestion,
    ) = route_metrics(
        graph,
        route,
    )

    fitness = fitness_function.calculate(
        travel_time,
        distance,
        congestion,
    )

    return {
        "algorithm": "Dijkstra",
        "route": route,
        "fitness": fitness,
        "travel_time": travel_time,
        "distance": distance,
        "congestion": congestion,
        "runtime": runtime,
        "iterations": 1,
    }


# ============================================================
# PSO
# ============================================================

def run_pso(
    graph,
    seed,
):
    optimizer = PSO(
        graph=graph,
        particles=PARTICLES,
        iterations=PSO_ITERATIONS,
        inertia=PSO_INERTIA,
        cognitive=PSO_COGNITIVE,
        social=PSO_SOCIAL,
        seed=seed,
        fitness_function=fitness_function,
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

    return {
        "algorithm": "PSO",
        "route": result["route"],
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
        "iterations": result["iterations"],
    }


# ============================================================
# QPSO
# ============================================================

def run_qpso(
    graph,
    seed,
):
    optimizer = QPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=QPSO_ITERATIONS,
        beta=QPSO_BETA,
        seed=seed,
        fitness_function=fitness_function,
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

    return {
        "algorithm": "QPSO",
        "route": result["route"],
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
        "iterations": result["iterations"],
    }


# ============================================================
# ADAPTIVE QPSO
# ============================================================

def run_adaptive(
    graph,
    seed,
):
    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ADAPTIVE_ITERATIONS,
        beta_max=BETA_MAX,
        beta_min=BETA_MIN,
        seed=seed,
        fitness_function=fitness_function,
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

    return {
        "algorithm": "Adaptive-QPSO",
        "route": result["route"],
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
        "iterations": result["iterations"],
    }


# ============================================================
# WARM ADAPTIVE QPSO
# ============================================================

def run_warm(
    graph,
    seed,
    initial_route,
):
    optimizer = WarmStartAdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=WARM_ITERATIONS,
        beta_max=BETA_MAX,
        beta_min=BETA_MIN,
        seed=seed,
        fitness_function=fitness_function,
        initial_route=initial_route,
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

    return {
        "algorithm": "Warm Adaptive-QPSO",
        "route": result["route"],
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
        "iterations": result["iterations"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    states = [
        "normal",
        "corridor_0",
        "corridor_1",
        "moving_corridor",
        "incident_50",
        "recovery",
    ]

    algorithms = [
        "Dijkstra",
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
        "Warm Adaptive-QPSO",
    ]

    raw_results = []

    print()
    print("=" * 110)
    print("SIH DYNAMIC DISRUPTION BENCHMARK")
    print("=" * 110)

    print(
        f"Network: {LAYERS} layers x {BRANCHES} branches"
    )

    print(
        f"Network seeds: {NETWORK_SEEDS}"
    )

    print(
        f"States: {', '.join(states)}"
    )

    print("=" * 110)

    # ========================================================
    # NETWORK SEEDS
    # ========================================================

    for network_seed in NETWORK_SEEDS:

        print()
        print(
            f"NETWORK SEED: {network_seed}"
        )

        previous_route = None

        # ====================================================
        # TRAFFIC STATES
        # ====================================================

        for state_index, state in enumerate(
            states
        ):

            graph = build_layered_graph(
                layers=LAYERS,
                branches=BRANCHES,
                seed=network_seed,
            )

            apply_state(
                graph,
                state,
            )

            algorithm_results = []

            # ------------------------------------------------
            # Dijkstra
            # ------------------------------------------------

            result = run_dijkstra(
                graph
            )

            algorithm_results.append(
                result
            )

            # ------------------------------------------------
            # PSO
            # ------------------------------------------------

            result = run_pso(
                graph,
                seed=(
                    ALGORITHM_SEED
                    + network_seed
                    + state_index
                ),
            )

            algorithm_results.append(
                result
            )

            # ------------------------------------------------
            # QPSO
            # ------------------------------------------------

            result = run_qpso(
                graph,
                seed=(
                    ALGORITHM_SEED
                    + network_seed
                    + state_index
                ),
            )

            algorithm_results.append(
                result
            )

            # ------------------------------------------------
            # Adaptive QPSO
            # ------------------------------------------------

            result = run_adaptive(
                graph,
                seed=(
                    ALGORITHM_SEED
                    + network_seed
                    + state_index
                ),
            )

            algorithm_results.append(
                result
            )

            # ------------------------------------------------
            # Warm-start Adaptive QPSO
            # ------------------------------------------------

            result = run_warm(
                graph,
                seed=(
                    ALGORITHM_SEED
                    + network_seed
                    + state_index
                ),
                initial_route=previous_route,
            )

            algorithm_results.append(
                result
            )

            # ------------------------------------------------
            # Update warm-start route
            # ------------------------------------------------

            previous_route = (
                algorithm_results[-1]["route"]
            )

            # ------------------------------------------------
            # Find best
            # ------------------------------------------------

            best = min(
                algorithm_results,
                key=lambda x: x["fitness"],
            )

            print(
                f"  {state:<18}"
                f" Best={best['algorithm']:<23}"
                f" Fitness={best['fitness']:.4f}"
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            for result in algorithm_results:

                raw_results.append(
                    {
                        "network_seed": network_seed,
                        "state": state,
                        "state_index": state_index,
                        "algorithm": result[
                            "algorithm"
                        ],
                        "fitness": result[
                            "fitness"
                        ],
                        "travel_time": result[
                            "travel_time"
                        ],
                        "distance": result[
                            "distance"
                        ],
                        "congestion": result[
                            "congestion"
                        ],
                        "runtime": result[
                            "runtime"
                        ],
                        "iterations": result[
                            "iterations"
                        ],
                        "route": "|".join(
                            str(node.id)
                            for node in result[
                                "route"
                            ]
                        ),
                    }
                )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    with open(
        RAW_RESULTS_FILE,
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "network_seed",
            "state",
            "state_index",
            "algorithm",
            "fitness",
            "travel_time",
            "distance",
            "congestion",
            "runtime",
            "iterations",
            "route",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            raw_results
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 110)
    print("OVERALL RESULTS")
    print("=" * 110)

    summary = []

    for algorithm in algorithms:

        rows = [
            row
            for row in raw_results
            if row["algorithm"]
            == algorithm
        ]

        fitness_values = [
            float(row["fitness"])
            for row in rows
        ]

        runtime_values = [
            float(row["runtime"])
            for row in rows
        ]

        mean_fitness = statistics.mean(
            fitness_values
        )

        std_fitness = statistics.stdev(
            fitness_values
        )

        mean_runtime = statistics.mean(
            runtime_values
        )

        # ----------------------------------------------------
        # Win count
        # ----------------------------------------------------

        wins = 0

        for seed in NETWORK_SEEDS:

            for state in states:

                candidates = [
                    row
                    for row in raw_results
                    if (
                        row["network_seed"]
                        == seed
                        and row["state"]
                        == state
                    )
                ]

                best_fitness = min(
                    float(row["fitness"])
                    for row in candidates
                )

                current = next(
                    row
                    for row in candidates
                    if row["algorithm"]
                    == algorithm
                )

                if abs(
                    float(current["fitness"])
                    - best_fitness
                ) < 1e-9:

                    wins += 1

        summary.append(
            {
                "algorithm": algorithm,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "mean_runtime": mean_runtime,
                "wins": wins,
            }
        )

        print(
            f"{algorithm:<25}"
            f" Mean Fitness={mean_fitness:<12.4f}"
            f" Std={std_fitness:<10.4f}"
            f" Runtime={mean_runtime:<10.5f}"
            f" Wins={wins}"
        )

    # ========================================================
    # SAVE OVERALL SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "mean_fitness",
                "std_fitness",
                "mean_runtime",
                "wins",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    # ========================================================
    # STATE-WISE SUMMARY
    # ========================================================

    state_summary = []

    print()
    print("=" * 110)
    print("STATE-WISE RESULTS")
    print("=" * 110)

    for state in states:

        print()
        print(
            f"STATE: {state}"
        )

        for algorithm in algorithms:

            rows = [
                row
                for row in raw_results
                if (
                    row["state"] == state
                    and row["algorithm"]
                    == algorithm
                )
            ]

            values = [
                float(row["fitness"])
                for row in rows
            ]

            runtimes = [
                float(row["runtime"])
                for row in rows
            ]

            mean_fitness = statistics.mean(
                values
            )

            mean_runtime = statistics.mean(
                runtimes
            )

            state_summary.append(
                {
                    "state": state,
                    "algorithm": algorithm,
                    "mean_fitness": mean_fitness,
                    "mean_runtime": mean_runtime,
                }
            )

            print(
                f"  {algorithm:<25}"
                f" Fitness={mean_fitness:<12.4f}"
                f" Runtime={mean_runtime:.5f}"
            )

    # ========================================================
    # SAVE STATE SUMMARY
    # ========================================================

    with open(
        STATE_SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "state",
                "algorithm",
                "mean_fitness",
                "mean_runtime",
            ],
        )

        writer.writeheader()

        writer.writerows(
            state_summary
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    best_algorithm = min(
        summary,
        key=lambda x: x["mean_fitness"],
    )

    fastest_algorithm = min(
        summary,
        key=lambda x: x["mean_runtime"],
    )

    most_wins_algorithm = max(
        summary,
        key=lambda x: x["wins"],
    )

    print()
    print("=" * 110)
    print("FINAL BENCHMARK CONCLUSION")
    print("=" * 110)

    print(
        f"Best mean fitness : "
        f"{best_algorithm['algorithm']}"
    )

    print(
        f"Mean fitness      : "
        f"{best_algorithm['mean_fitness']:.6f}"
    )

    print(
        f"Fastest algorithm : "
        f"{fastest_algorithm['algorithm']}"
    )

    print(
        f"Most wins         : "
        f"{most_wins_algorithm['algorithm']}"
        f" ({most_wins_algorithm['wins']}/"
        f"{len(NETWORK_SEEDS) * len(states)})"
    )

    print()
    print(
        "Files generated:"
    )

    print(
        f"  {RAW_RESULTS_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {STATE_SUMMARY_FILE}"
    )

    print("=" * 110)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
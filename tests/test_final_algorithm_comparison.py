"""
Final Algorithm Comparison Benchmark

Algorithms:
    1. Dijkstra
    2. PSO
    3. Fixed-QPSO
    4. Adaptive-QPSO
    5. Warm-Start Adaptive-QPSO

Traffic states:
    - normal
    - gradual_25
    - gradual_50
    - incident_50
    - moving_70
    - normal_recovery

Purpose:
    Compare classical shortest-path optimization with
    quantum-inspired metaheuristic approaches under
    controlled dynamic traffic conditions.

Important:
    Dijkstra minimizes travel time only.
    PSO/QPSO optimize the common multi-objective
    FitnessFunction.
"""

import csv
import os
import random
import time

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

# Warm-start intentionally uses fewer iterations.
WARM_ITERATIONS = 30

PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.5
PSO_SOCIAL = 1.5

QPSO_BETA = 0.9

BETA_MAX = 0.9
BETA_MIN = 0.5

NETWORK_SEED = 42
ALGORITHM_SEED = 42

START = "N0"
DESTINATION = f"N{LAYERS + 1}"

RESULTS_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "final_algorithm_comparison.csv",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "final_algorithm_summary.csv",
)


# ============================================================
# FITNESS FUNCTION
# ============================================================

fitness_function = FitnessFunction(
    time_weight=0.5,
    distance_weight=0.3,
    congestion_weight=0.2,
)


# ============================================================
# GRAPH GENERATION
# ============================================================

def build_layered_graph(
    layers=LAYERS,
    branches=BRANCHES,
    seed=NETWORK_SEED,
):
    """
    Create a controlled layered transportation graph.

    Structure:

        N0
         |
       L1
      / | \
     ...
       L100
         |
       N101

    Each node in a layer connects to every node
    in the next layer.
    """

    graph = Graph()

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Start node
    # --------------------------------------------------------

    start_node = Node("N0")
    graph.add_node(start_node)

    # Store layer nodes.
    layer_nodes = {}

    # --------------------------------------------------------
    # Create layers
    # --------------------------------------------------------

    node_counter = 1

    for layer in range(1, layers + 1):

        current_layer = []

        for branch in range(branches):

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
    # Start -> first layer
    # --------------------------------------------------------

    for node in layer_nodes[1]:

        distance = 5.0
        congestion = 0.10

        travel_time = (
            distance
            * (1.0 + congestion)
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

        current_layer = layer_nodes[layer]

        next_layer = layer_nodes[
            layer + 1
        ]

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

    for node in layer_nodes[layers]:

        distance = 5.0
        congestion = 0.10

        travel_time = (
            distance
            * (1.0 + congestion)
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
# TRAFFIC STATE
# ============================================================

def apply_traffic_state(
    graph,
    state,
):
    """
    Apply one controlled traffic scenario.
    """

    traffic = DynamicTrafficModel(
        base_congestion=0.10,
        congestion_factor=5.0,
    )

    # --------------------------------------------------------
    # Normal traffic
    # --------------------------------------------------------

    if state == "normal":

        traffic.normal(
            graph
        )

    # --------------------------------------------------------
    # Gradual congestion 25%
    # --------------------------------------------------------

    elif state == "gradual_25":

        traffic.gradual_congestion(
            graph,
            level=0.25,
        )

    # --------------------------------------------------------
    # Gradual congestion 50%
    # --------------------------------------------------------

    elif state == "gradual_50":

        traffic.gradual_congestion(
            graph,
            level=0.50,
        )

    # --------------------------------------------------------
    # Incident around layer 50
    # --------------------------------------------------------

    elif state == "incident_50":

        traffic.incident(
            graph,
            layer=50,
            incident_congestion=1.0,
        )

    # --------------------------------------------------------
    # Moving congestion around layer 70
    # --------------------------------------------------------

    elif state == "moving_70":

        traffic.moving_congestion(
            graph,
            layer=70,
            congestion=0.90,
        )

    # --------------------------------------------------------
    # Recovery to normal
    # --------------------------------------------------------

    elif state == "normal_recovery":

        traffic.normal(
            graph
        )

    else:

        raise ValueError(
            f"Unknown traffic state: {state}"
        )


# ============================================================
# ROUTE METRICS
# ============================================================

def calculate_route_metrics(
    graph,
    route,
):
    """
    Calculate:

        total travel time
        total distance
        average congestion
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

        edge_found = None

        for edge in graph.get_neighbors(
            source
        ):

            if (
                edge.target.id
                == target.id
                and not edge.blocked
            ):

                edge_found = edge
                break

        if edge_found is None:

            return (
                float("inf"),
                float("inf"),
                1.0,
            )

        total_time += (
            edge_found.travel_time
        )

        total_distance += (
            edge_found.distance
        )

        congestion_values.append(
            edge_found.congestion
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


# ============================================================
# ROUTE FITNESS
# ============================================================

def calculate_fitness(
    travel_time,
    distance,
    congestion,
):
    """
    Evaluate a route using the common
    multi-objective fitness function.
    """

    return fitness_function.calculate(
        travel_time=travel_time,
        distance=distance,
        congestion=congestion,
    )


# ============================================================
# DIJKSTRA
# ============================================================

def run_dijkstra(
    graph,
):
    """
    Run classical Dijkstra.

    Dijkstra optimizes travel time only.
    The resulting route is then evaluated
    using the common FitnessFunction.
    """

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

    # --------------------------------------------------------
    # Use Dijkstra's returned metrics if available.
    # Otherwise calculate them directly.
    # --------------------------------------------------------

    travel_time = result.get(
        "travel_time"
    )

    distance = result.get(
        "distance"
    )

    congestion = result.get(
        "congestion"
    )

    if (
        travel_time is None
        or distance is None
        or congestion is None
    ):

        (
            travel_time,
            distance,
            congestion,
        ) = calculate_route_metrics(
            graph,
            route,
        )

    # --------------------------------------------------------
    # Use returned fitness if available.
    # Otherwise evaluate using common fitness.
    # --------------------------------------------------------

    fitness = result.get(
        "fitness"
    )

    if fitness is None:

        fitness = calculate_fitness(
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
    """
    Run classical Particle Swarm Optimization.
    """

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
# FIXED QPSO
# ============================================================

def run_qpso(
    graph,
    seed,
):
    """
    Run fixed-beta QPSO.

    beta = 0.9
    """

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

def run_adaptive_qpso(
    graph,
    seed,
):
    """
    Run Adaptive-QPSO.

    beta:
        0.9 -> 0.5
    """

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
# WARM-START ADAPTIVE QPSO
# ============================================================

def run_warm_adaptive_qpso(
    graph,
    seed,
    initial_route,
):
    """
    Run Warm-Start Adaptive-QPSO.

    beta:
        0.9 -> 0.5

    Uses only 30 iterations instead of 50.
    """

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
# ROUTE SIGNATURE
# ============================================================

def route_signature(
    route,
):
    """
    Convert a route into a comparable tuple.
    """

    if route is None:
        return None

    return tuple(
        str(node.id)
        for node in route
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result,
):
    """
    Print one algorithm result.
    """

    print(
        f"{result['algorithm']:<22}"
        f" Fitness: {result['fitness']:<12.6f}"
        f" Runtime: {result['runtime']:<10.6f}"
        f" Time: {result['travel_time']:<12.6f}"
        f" Distance: {result['distance']:<12.6f}"
        f" Congestion: {result['congestion']:.6f}"
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    traffic_states = [
        "normal",
        "gradual_25",
        "gradual_50",
        "incident_50",
        "moving_70",
        "normal_recovery",
    ]

    all_results = []

    previous_warm_route = None

    print()
    print("=" * 90)
    print("FINAL ALGORITHM COMPARISON")
    print("=" * 90)

    print(
        "Dijkstra              : classical shortest travel-time baseline"
    )

    print(
        "PSO                   : 50 iterations"
    )

    print(
        "QPSO                  : beta = 0.9, 50 iterations"
    )

    print(
        "Adaptive-QPSO         : beta = 0.9 -> 0.5, 50 iterations"
    )

    print(
        "Warm Adaptive-QPSO   : beta = 0.9 -> 0.5, 30 iterations"
    )

    print(
        "Traffic states        : "
        + ", ".join(traffic_states)
    )

    print("=" * 90)

    # ========================================================
    # TRAFFIC STATES
    # ========================================================

    for state_index, state in enumerate(
        traffic_states
    ):

        print()
        print("=" * 90)
        print(
            f"TRAFFIC STATE: {state}"
        )
        print("=" * 90)

        # ----------------------------------------------------
        # Create a fresh identical graph.
        # ----------------------------------------------------

        graph = build_layered_graph(
            layers=LAYERS,
            branches=BRANCHES,
            seed=NETWORK_SEED,
        )

        # ----------------------------------------------------
        # Apply traffic condition.
        # ----------------------------------------------------

        apply_traffic_state(
            graph,
            state,
        )

        # ====================================================
        # DIJKSTRA
        # ====================================================

        dijkstra_result = run_dijkstra(
            graph
        )

        print_result(
            dijkstra_result
        )

        all_results.append(
            {
                "state": state,
                "state_index": state_index,
                **dijkstra_result,
            }
        )

        # ====================================================
        # PSO
        # ====================================================

        pso_result = run_pso(
            graph,
            seed=ALGORITHM_SEED + state_index,
        )

        print_result(
            pso_result
        )

        all_results.append(
            {
                "state": state,
                "state_index": state_index,
                **pso_result,
            }
        )

        # ====================================================
        # QPSO
        # ====================================================

        qpso_result = run_qpso(
            graph,
            seed=ALGORITHM_SEED + state_index,
        )

        print_result(
            qpso_result
        )

        all_results.append(
            {
                "state": state,
                "state_index": state_index,
                **qpso_result,
            }
        )

        # ====================================================
        # ADAPTIVE QPSO
        # ====================================================

        adaptive_result = run_adaptive_qpso(
            graph,
            seed=ALGORITHM_SEED + state_index,
        )

        print_result(
            adaptive_result
        )

        all_results.append(
            {
                "state": state,
                "state_index": state_index,
                **adaptive_result,
            }
        )

        # ====================================================
        # WARM-START ADAPTIVE QPSO
        # ====================================================

        warm_result = run_warm_adaptive_qpso(
            graph,
            seed=ALGORITHM_SEED + state_index,
            initial_route=previous_warm_route,
        )

        print_result(
            warm_result
        )

        all_results.append(
            {
                "state": state,
                "state_index": state_index,
                **warm_result,
            }
        )

        # ----------------------------------------------------
        # Update warm-start route.
        #
        # The current optimized route becomes the initial
        # route for the next traffic state.
        # ----------------------------------------------------

        previous_warm_route = (
            warm_result["route"]
        )

        # ====================================================
        # BEST ALGORITHM
        # ====================================================

        state_results = [
            dijkstra_result,
            pso_result,
            qpso_result,
            adaptive_result,
            warm_result,
        ]

        best = min(
            state_results,
            key=lambda x: x["fitness"],
        )

        print()
        print(
            f"BEST: {best['algorithm']}"
            f" | Fitness = {best['fitness']:.6f}"
        )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "state",
                "state_index",
                "algorithm",
                "fitness",
                "travel_time",
                "distance",
                "congestion",
                "runtime",
                "iterations",
            ],
        )

        writer.writeheader()

        for result in all_results:

            writer.writerow(
                {
                    "state": result["state"],
                    "state_index": result["state_index"],
                    "algorithm": result["algorithm"],
                    "fitness": result["fitness"],
                    "travel_time": result["travel_time"],
                    "distance": result["distance"],
                    "congestion": result["congestion"],
                    "runtime": result["runtime"],
                    "iterations": result["iterations"],
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    algorithms = [
        "Dijkstra",
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
        "Warm Adaptive-QPSO",
    ]

    summary = []

    print()
    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    print(
        f"{'Algorithm':<25}"
        f"{'Mean Fitness':<18}"
        f"{'Mean Runtime':<18}"
        f"{'Wins':<10}"
        f"{'Iterations':<12}"
    )

    print("-" * 100)

    # --------------------------------------------------------
    # Calculate mean metrics.
    # --------------------------------------------------------

    for algorithm in algorithms:

        algorithm_results = [
            result
            for result in all_results
            if result["algorithm"]
            == algorithm
        ]

        mean_fitness = (
            sum(
                result["fitness"]
                for result in algorithm_results
            )
            / len(algorithm_results)
        )

        mean_runtime = (
            sum(
                result["runtime"]
                for result in algorithm_results
            )
            / len(algorithm_results)
        )

        iterations = (
            sum(
                result["iterations"]
                for result in algorithm_results
            )
            / len(algorithm_results)
        )

        wins = 0

        for state in traffic_states:

            state_results = [
                result
                for result in all_results
                if (
                    result["state"] == state
                    and result["algorithm"]
                    in algorithms
                )
            ]

            best_fitness = min(
                result["fitness"]
                for result in state_results
            )

            current = next(
                result
                for result in state_results
                if result["algorithm"]
                == algorithm
            )

            if (
                abs(
                    current["fitness"]
                    - best_fitness
                )
                < 1e-9
            ):

                wins += 1

        summary_row = {
            "algorithm": algorithm,
            "mean_fitness": mean_fitness,
            "mean_runtime": mean_runtime,
            "wins": wins,
            "mean_iterations": iterations,
        }

        summary.append(
            summary_row
        )

        print(
            f"{algorithm:<25}"
            f"{mean_fitness:<18.6f}"
            f"{mean_runtime:<18.6f}"
            f"{wins:<10}"
            f"{iterations:<12.2f}"
        )

    # ========================================================
    # SAVE SUMMARY
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
                "mean_runtime",
                "wins",
                "mean_iterations",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    # ========================================================
    # BEST OVERALL
    # ========================================================

    best_overall = min(
        summary,
        key=lambda x: x["mean_fitness"],
    )

    fastest = min(
        summary,
        key=lambda x: x["mean_runtime"],
    )

    most_wins = max(
        summary,
        key=lambda x: x["wins"],
    )

    print()
    print("=" * 100)
    print("FINAL CONCLUSIONS")
    print("=" * 100)

    print(
        f"Best mean fitness : "
        f"{best_overall['algorithm']}"
        f" ({best_overall['mean_fitness']:.6f})"
    )

    print(
        f"Fastest runtime   : "
        f"{fastest['algorithm']}"
        f" ({fastest['mean_runtime']:.6f} s)"
    )

    print(
        f"Most state wins   : "
        f"{most_wins['algorithm']}"
        f" ({most_wins['wins']}/{len(traffic_states)})"
    )

    print()
    print(
        f"Raw results saved to:"
    )
    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to:"
    )
    print(
        f"  {SUMMARY_FILE}"
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
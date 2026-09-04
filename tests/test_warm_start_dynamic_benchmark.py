import csv
import time
import random

from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.traffic.traffic_model import DynamicTrafficModel

from src.algorithms.adaptive_qpso import AdaptiveQPSO
from src.algorithms.warm_start_adaptive_qpso import WarmStartAdaptiveQPSO


SEED = 42

LAYERS = 100
BRANCHES = 3

PARTICLES = 20

ITERATION_BUDGETS = [
    10,
    20,
    30,
    50,
]

START = "N0"
DESTINATION = "N100"


# ============================================================
# Graph generation
# ============================================================

def create_graph(seed=42):

    rng = random.Random(seed)

    graph = Graph()

    start = Node("N0")
    destination = Node("N100")

    layer_nodes = {}

    # --------------------------------------------------------
    # Internal layers
    # --------------------------------------------------------

    for layer in range(1, LAYERS):

        layer_nodes[layer] = []

        for branch in range(BRANCHES):

            node = Node(
                f"L{layer}_B{branch}"
            )

            layer_nodes[layer].append(node)

    # --------------------------------------------------------
    # Start -> first layer
    # --------------------------------------------------------

    for node in layer_nodes[1]:

        distance = rng.uniform(4.0, 6.0)

        congestion = 0.10

        travel_time = (
            distance
            * (1.0 + congestion)
        )

        graph.add_edge(
            Edge(
                start,
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
        LAYERS - 1,
    ):

        current_nodes = (
            layer_nodes[layer]
        )

        next_nodes = (
            layer_nodes[layer + 1]
        )

        for source in current_nodes:

            for target in next_nodes:

                distance = rng.uniform(
                    4.0,
                    6.0,
                )

                congestion = 0.10

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
    # Final layer -> destination
    # --------------------------------------------------------

    for source in layer_nodes[
        LAYERS - 1
    ]:

        distance = rng.uniform(
            4.0,
            6.0,
        )

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
# Traffic scenarios
# ============================================================

def apply_traffic_state(
    graph,
    traffic_model,
    state,
):

    if state == "normal":

        traffic_model.normal(
            graph
        )

    elif state == "gradual_25":

        traffic_model.gradual_congestion(
            graph,
            0.25,
        )

    elif state == "gradual_50":

        traffic_model.gradual_congestion(
            graph,
            0.50,
        )

    elif state == "incident_50":

        traffic_model.normal(
            graph
        )

        traffic_model.incident(
            graph,
            layer=50,
        )

    elif state == "moving_70":

        traffic_model.normal(
            graph
        )

        traffic_model.moving_congestion(
            graph,
            layer=70,
            congestion=0.90,
        )

    elif state == "normal_recovery":

        traffic_model.normal(
            graph
        )

    else:

        raise ValueError(
            f"Unknown state: {state}"
        )


# ============================================================
# Cold start
# ============================================================

def run_cold_start(
    graph,
    iterations,
    seed,
):

    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=iterations,
        beta_max=0.9,
        beta_min=0.5,
        seed=seed,
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


# ============================================================
# Warm start
# ============================================================

def run_warm_start(
    graph,
    iterations,
    seed,
    previous_route,
):

    optimizer = (
        WarmStartAdaptiveQPSO(
            graph=graph,
            particles=PARTICLES,
            iterations=iterations,
            beta_max=0.9,
            beta_min=0.5,
            seed=seed,
            initial_route=previous_route,
            initialization_attempts=100,
        )
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


# ============================================================
# Route helper
# ============================================================

def route_to_ids(route):

    if route is None:
        return []

    return [
        node.id
        for node in route
    ]


# ============================================================
# Main benchmark
# ============================================================

def main():

    print(
        "\n"
        + "=" * 95
    )

    print(
        "COLD START vs WARM START ADAPTIVE-QPSO"
    )

    print(
        "=" * 95
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

    # --------------------------------------------------------
    # Iterate over optimization budgets
    # --------------------------------------------------------

    for iterations in ITERATION_BUDGETS:

        print(
            f"\nITERATION BUDGET: "
            f"{iterations}"
        )

        print(
            "-" * 95
        )

        print(
            f"{'Traffic State':<18}"
            f"{'Cold Fitness':>15}"
            f"{'Warm Fitness':>15}"
            f"{'Cold Time':>13}"
            f"{'Warm Time':>13}"
            f"{'Improvement':>14}"
        )

        print(
            "-" * 95
        )

        # ----------------------------------------------------
        # Separate persistent graphs
        # ----------------------------------------------------

        cold_graph = create_graph(
            SEED
        )

        warm_graph = create_graph(
            SEED
        )

        cold_traffic = (
            DynamicTrafficModel(
                seed=SEED
            )
        )

        warm_traffic = (
            DynamicTrafficModel(
                seed=SEED
            )
        )

        previous_warm_route = None

        # ----------------------------------------------------
        # Dynamic traffic sequence
        # ----------------------------------------------------

        for time_step, state in enumerate(
            traffic_states
        ):

            apply_traffic_state(
                cold_graph,
                cold_traffic,
                state,
            )

            apply_traffic_state(
                warm_graph,
                warm_traffic,
                state,
            )

            run_seed = (
                SEED
                + iterations
                + time_step
            )

            # ================================================
            # Cold start
            # ================================================

            cold_result, cold_runtime = (
                run_cold_start(
                    cold_graph,
                    iterations,
                    run_seed,
                )
            )

            # ================================================
            # Warm start
            # ================================================

            warm_result, warm_runtime = (
                run_warm_start(
                    warm_graph,
                    iterations,
                    run_seed,
                    previous_warm_route,
                )
            )

            # Store route for next traffic state
            previous_warm_route = list(
                warm_result["route"]
            )

            # ================================================
            # Metrics
            # ================================================

            cold_fitness = (
                cold_result["fitness"]
            )

            warm_fitness = (
                warm_result["fitness"]
            )

            if cold_fitness != 0:

                improvement = (
                    (
                        cold_fitness
                        - warm_fitness
                    )
                    / cold_fitness
                    * 100.0
                )

            else:

                improvement = 0.0

            cold_route_ids = (
                route_to_ids(
                    cold_result["route"]
                )
            )

            warm_route_ids = (
                route_to_ids(
                    warm_result["route"]
                )
            )

            routes_equal = (
                cold_route_ids
                == warm_route_ids
            )

            # ================================================
            # Output
            # ================================================

            print(
                f"{state:<18}"
                f"{cold_fitness:>15.6f}"
                f"{warm_fitness:>15.6f}"
                f"{cold_runtime:>13.6f}"
                f"{warm_runtime:>13.6f}"
                f"{improvement:>13.2f}%"
            )

            all_results.append(
                {
                    "iterations": iterations,
                    "time_step": time_step,
                    "traffic_state": state,

                    "cold_fitness":
                        cold_fitness,

                    "warm_fitness":
                        warm_fitness,

                    "cold_runtime":
                        cold_runtime,

                    "warm_runtime":
                        warm_runtime,

                    "fitness_improvement_percent":
                        improvement,

                    "cold_travel_time":
                        cold_result[
                            "travel_time"
                        ],

                    "warm_travel_time":
                        warm_result[
                            "travel_time"
                        ],

                    "cold_distance":
                        cold_result[
                            "distance"
                        ],

                    "warm_distance":
                        warm_result[
                            "distance"
                        ],

                    "cold_congestion":
                        cold_result[
                            "congestion"
                        ],

                    "warm_congestion":
                        warm_result[
                            "congestion"
                        ],

                    "warm_started":
                        warm_result[
                            "warm_started"
                        ],

                    "routes_equal":
                        routes_equal,
                }
            )

    # ========================================================
    # Save detailed CSV
    # ========================================================

    output_file = (
        "results/"
        "warm_start_dynamic_benchmark.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
    ) as csvfile:

        fieldnames = list(
            all_results[0].keys()
        )

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "SUMMARY BY ITERATION BUDGET"
    )

    print(
        "=" * 95
    )

    print(
        f"{'Iterations':<12}"
        f"{'Cold Mean':>14}"
        f"{'Warm Mean':>14}"
        f"{'Fitness Gain':>15}"
        f"{'Cold Runtime':>15}"
        f"{'Warm Runtime':>15}"
    )

    print(
        "-" * 95
    )

    for iterations in ITERATION_BUDGETS:

        subset = [
            row
            for row in all_results
            if row["iterations"]
            == iterations
        ]

        cold_mean = sum(
            row["cold_fitness"]
            for row in subset
        ) / len(subset)

        warm_mean = sum(
            row["warm_fitness"]
            for row in subset
        ) / len(subset)

        cold_runtime_mean = sum(
            row["cold_runtime"]
            for row in subset
        ) / len(subset)

        warm_runtime_mean = sum(
            row["warm_runtime"]
            for row in subset
        ) / len(subset)

        fitness_gain = (
            (
                cold_mean
                - warm_mean
            )
            / cold_mean
            * 100.0
        )

        print(
            f"{iterations:<12}"
            f"{cold_mean:>14.6f}"
            f"{warm_mean:>14.6f}"
            f"{fitness_gain:>14.2f}%"
            f"{cold_runtime_mean:>15.6f}"
            f"{warm_runtime_mean:>15.6f}"
        )

    # ========================================================
    # Warm-start wins
    # ========================================================

    warm_wins = sum(
        1
        for row in all_results
        if row["warm_fitness"]
        < row["cold_fitness"]
    )

    cold_wins = sum(
        1
        for row in all_results
        if row["cold_fitness"]
        < row["warm_fitness"]
    )

    equal = (
        len(all_results)
        - warm_wins
        - cold_wins
    )

    print(
        "\nOverall comparisons:"
    )

    print(
        f"  Total comparisons : "
        f"{len(all_results)}"
    )

    print(
        f"  Warm-start wins   : "
        f"{warm_wins}"
    )

    print(
        f"  Cold-start wins   : "
        f"{cold_wins}"
    )

    print(
        f"  Equal             : "
        f"{equal}"
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        f"  {output_file}"
    )

    print(
        "\nBenchmark completed."
    )


if __name__ == "__main__":
    main()
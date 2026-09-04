import csv
import time
import random

from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.traffic.traffic_model import DynamicTrafficModel

from src.algorithms.adaptive_qpso import AdaptiveQPSO
from src.algorithms.warm_start_adaptive_qpso import (
    WarmStartAdaptiveQPSO,
)


SEED = 42

LAYERS = 100
BRANCHES = 3

PARTICLES = 20

COLD_ITERATIONS = 50

WARM_ITERATIONS = [
    10,
    20,
    30,
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

    # Internal layers
    for layer in range(1, LAYERS):

        layer_nodes[layer] = []

        for branch in range(BRANCHES):

            node = Node(
                f"L{layer}_B{branch}"
            )

            layer_nodes[layer].append(node)

    # Start -> first layer
    for node in layer_nodes[1]:

        distance = rng.uniform(4.0, 6.0)
        congestion = 0.10

        travel_time = (
            distance * (1.0 + congestion)
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

    # Internal layers
    for layer in range(1, LAYERS - 1):

        current_nodes = layer_nodes[layer]
        next_nodes = layer_nodes[layer + 1]

        for source in current_nodes:

            for target in next_nodes:

                distance = rng.uniform(4.0, 6.0)
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

    # Final layer -> destination
    for source in layer_nodes[LAYERS - 1]:

        distance = rng.uniform(4.0, 6.0)
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
# Traffic state
# ============================================================

def apply_traffic_state(
    graph,
    traffic_model,
    state,
):

    if state == "normal":

        traffic_model.normal(graph)

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

        traffic_model.normal(graph)

        traffic_model.incident(
            graph,
            layer=50,
        )

    elif state == "moving_70":

        traffic_model.normal(graph)

        traffic_model.moving_congestion(
            graph,
            layer=70,
            congestion=0.90,
        )

    elif state == "normal_recovery":

        traffic_model.normal(graph)

    else:

        raise ValueError(
            f"Unknown traffic state: {state}"
        )


# ============================================================
# Cold-start optimizer
# ============================================================

def run_cold(
    graph,
    seed,
):

    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=COLD_ITERATIONS,
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
# Warm-start optimizer
# ============================================================

def run_warm(
    graph,
    iterations,
    seed,
    previous_route,
):

    optimizer = WarmStartAdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=iterations,
        beta_max=0.9,
        beta_min=0.5,
        seed=seed,
        initial_route=previous_route,
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
# Main
# ============================================================

def main():

    print("\n" + "=" * 100)

    print(
        "REDUCED-ITERATION WARM-START BENCHMARK"
    )

    print("=" * 100)

    print(
        "\nCold-start reference: "
        f"{COLD_ITERATIONS} iterations"
    )

    print(
        "Warm-start budgets: "
        "10, 20, 30 iterations"
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

    # ========================================================
    # Run each warm iteration budget
    # ========================================================

    for warm_iterations in WARM_ITERATIONS:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"WARM ITERATIONS: "
            f"{warm_iterations}"
        )

        print("=" * 100)

        print(
            f"{'Traffic State':<18}"
            f"{'Cold 50':>15}"
            f"{'Warm':>15}"
            f"{'Cold Time':>13}"
            f"{'Warm Time':>13}"
            f"{'Fitness Diff':>15}"
            f"{'Runtime Save':>15}"
        )

        print("-" * 100)

        # Persistent networks
        cold_graph = create_graph(SEED)
        warm_graph = create_graph(SEED)

        cold_traffic = DynamicTrafficModel(
            seed=SEED
        )

        warm_traffic = DynamicTrafficModel(
            seed=SEED
        )

        previous_route = None

        # ====================================================
        # Dynamic traffic sequence
        # ====================================================

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
                + warm_iterations
                + time_step
            )

            # ------------------------------------------------
            # Cold
            # ------------------------------------------------

            cold_result, cold_runtime = run_cold(
                cold_graph,
                run_seed,
            )

            # ------------------------------------------------
            # Warm
            # ------------------------------------------------

            warm_result, warm_runtime = run_warm(
                warm_graph,
                warm_iterations,
                run_seed,
                previous_route,
            )

            # Store route for next traffic state
            previous_route = list(
                warm_result["route"]
            )

            # ------------------------------------------------
            # Fitness
            # ------------------------------------------------

            cold_fitness = (
                cold_result["fitness"]
            )

            warm_fitness = (
                warm_result["fitness"]
            )

            fitness_difference = (
                (
                    warm_fitness
                    - cold_fitness
                )
                / cold_fitness
                * 100.0
            )

            # ------------------------------------------------
            # Runtime saving
            # ------------------------------------------------

            runtime_saving = (
                (
                    cold_runtime
                    - warm_runtime
                )
                / cold_runtime
                * 100.0
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"{state:<18}"
                f"{cold_fitness:>15.6f}"
                f"{warm_fitness:>15.6f}"
                f"{cold_runtime:>13.6f}"
                f"{warm_runtime:>13.6f}"
                f"{fitness_difference:>14.2f}%"
                f"{runtime_saving:>14.2f}%"
            )

            all_results.append(
                {
                    "warm_iterations":
                        warm_iterations,

                    "cold_iterations":
                        COLD_ITERATIONS,

                    "time_step":
                        time_step,

                    "traffic_state":
                        state,

                    "cold_fitness":
                        cold_fitness,

                    "warm_fitness":
                        warm_fitness,

                    "cold_runtime":
                        cold_runtime,

                    "warm_runtime":
                        warm_runtime,

                    "fitness_difference_percent":
                        fitness_difference,

                    "runtime_saving_percent":
                        runtime_saving,

                    "warm_started":
                        warm_result[
                            "warm_started"
                        ],
                }
            )

    # ========================================================
    # Save CSV
    # ========================================================

    output_file = (
        "results/"
        "warm_start_reduced_iterations.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
    ) as file:

        fieldnames = list(
            all_results[0].keys()
        )

        writer = csv.DictWriter(
            file,
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
        + "=" * 100
    )

    print(
        "SUMMARY"
    )

    print("=" * 100)

    print(
        f"{'Warm Iterations':<18}"
        f"{'Cold Mean':>15}"
        f"{'Warm Mean':>15}"
        f"{'Fitness Gap':>15}"
        f"{'Cold Time':>15}"
        f"{'Warm Time':>15}"
        f"{'Runtime Save':>15}"
    )

    print("-" * 100)

    for warm_iterations in WARM_ITERATIONS:

        subset = [
            row
            for row in all_results
            if row["warm_iterations"]
            == warm_iterations
        ]

        cold_mean = sum(
            row["cold_fitness"]
            for row in subset
        ) / len(subset)

        warm_mean = sum(
            row["warm_fitness"]
            for row in subset
        ) / len(subset)

        cold_time = sum(
            row["cold_runtime"]
            for row in subset
        ) / len(subset)

        warm_time = sum(
            row["warm_runtime"]
            for row in subset
        ) / len(subset)

        fitness_gap = (
            (
                warm_mean
                - cold_mean
            )
            / cold_mean
            * 100.0
        )

        runtime_save = (
            (
                cold_time
                - warm_time
            )
            / cold_time
            * 100.0
        )

        print(
            f"{warm_iterations:<18}"
            f"{cold_mean:>15.6f}"
            f"{warm_mean:>15.6f}"
            f"{fitness_gap:>14.2f}%"
            f"{cold_time:>15.6f}"
            f"{warm_time:>15.6f}"
            f"{runtime_save:>14.2f}%"
        )

    # ========================================================
    # Overall statistics
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "OVERALL RESULT"
    )

    print("=" * 100)

    for warm_iterations in WARM_ITERATIONS:

        subset = [
            row
            for row in all_results
            if row["warm_iterations"]
            == warm_iterations
        ]

        warm_wins = sum(
            1
            for row in subset
            if row["warm_fitness"]
            <= row["cold_fitness"]
        )

        avg_fitness_gap = sum(
            row[
                "fitness_difference_percent"
            ]
            for row in subset
        ) / len(subset)

        avg_runtime_save = sum(
            row[
                "runtime_saving_percent"
            ]
            for row in subset
        ) / len(subset)

        print(
            f"\nWarm {warm_iterations} "
            f"vs Cold {COLD_ITERATIONS}:"
        )

        print(
            f"  Warm-start wins/equal : "
            f"{warm_wins}/{len(subset)}"
        )

        print(
            f"  Average fitness gap   : "
            f"{avg_fitness_gap:.2f}%"
        )

        print(
            f"  Average runtime save  : "
            f"{avg_runtime_save:.2f}%"
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
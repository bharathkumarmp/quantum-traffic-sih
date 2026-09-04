import csv
import math
import random
import statistics
import time

from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.traffic.traffic_model import DynamicTrafficModel

from src.algorithms.adaptive_qpso import AdaptiveQPSO
from src.algorithms.warm_start_adaptive_qpso import (
    WarmStartAdaptiveQPSO,
)


# ============================================================
# Benchmark configuration
# ============================================================

NETWORK_SEEDS = list(range(42, 52))

LAYERS = 100
BRANCHES = 3

PARTICLES = 20

COLD_ITERATIONS = 50
WARM_ITERATIONS = 30

START = "N0"
DESTINATION = "N100"


TRAFFIC_STATES = [
    "normal",
    "gradual_25",
    "gradual_50",
    "incident_50",
    "moving_70",
    "normal_recovery",
]


# ============================================================
# Graph generation
# ============================================================

def create_graph(seed):

    rng = random.Random(seed)

    graph = Graph()

    start = Node("N0")
    destination = Node("N100")

    layer_nodes = {}

    # --------------------------------------------------------
    # Create internal layers
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
                start,
                node,
                distance=distance,
                travel_time=travel_time,
                congestion=congestion,
            )
        )

    # --------------------------------------------------------
    # Internal connections
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
    # Last layer -> destination
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
# Traffic state
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
            f"Unknown traffic state: {state}"
        )


# ============================================================
# Cold-start Adaptive-QPSO
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
# Warm-start Adaptive-QPSO
# ============================================================

def run_warm(
    graph,
    seed,
    previous_route,
):

    optimizer = WarmStartAdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=WARM_ITERATIONS,
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
# Main benchmark
# ============================================================

def main():

    print("\n" + "=" * 110)

    print(
        "MULTI-SEED WARM-START STATISTICAL VALIDATION"
    )

    print("=" * 110)

    print(
        f"\nNetworks       : {len(NETWORK_SEEDS)}"
    )

    print(
        f"Traffic states : {len(TRAFFIC_STATES)}"
    )

    print(
        f"Cold iterations: {COLD_ITERATIONS}"
    )

    print(
        f"Warm iterations: {WARM_ITERATIONS}"
    )

    print(
        f"Particles      : {PARTICLES}"
    )

    print(
        f"Total pairs    : "
        f"{len(NETWORK_SEEDS) * len(TRAFFIC_STATES)}"
    )

    all_results = []

    # ========================================================
    # Run every network seed
    # ========================================================

    for network_seed in NETWORK_SEEDS:

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"NETWORK SEED: {network_seed}"
        )

        print("=" * 110)

        # Same topology for cold and warm experiments
        cold_graph = create_graph(
            network_seed
        )

        warm_graph = create_graph(
            network_seed
        )

        cold_traffic = DynamicTrafficModel(
            seed=network_seed
        )

        warm_traffic = DynamicTrafficModel(
            seed=network_seed
        )

        previous_warm_route = None

        # ----------------------------------------------------
        # Traffic sequence
        # ----------------------------------------------------

        for time_step, state in enumerate(
            TRAFFIC_STATES
        ):

            print(
                f"  {state:<18}",
                end="",
                flush=True,
            )

            # Apply exactly the same traffic state
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

            # ------------------------------------------------
            # Independent but paired optimizer seed
            # ------------------------------------------------

            optimizer_seed = (
                network_seed * 100
                + time_step
            )

            # ------------------------------------------------
            # Cold start
            # ------------------------------------------------

            cold_result, cold_runtime = (
                run_cold(
                    cold_graph,
                    optimizer_seed,
                )
            )

            # ------------------------------------------------
            # Warm start
            # ------------------------------------------------

            warm_result, warm_runtime = (
                run_warm(
                    warm_graph,
                    optimizer_seed,
                    previous_warm_route,
                )
            )

            # ------------------------------------------------
            # Previous warm route becomes next initial route
            # ------------------------------------------------

            previous_warm_route = list(
                warm_result["route"]
            )

            # ------------------------------------------------
            # Fitness comparison
            # ------------------------------------------------

            cold_fitness = (
                cold_result["fitness"]
            )

            warm_fitness = (
                warm_result["fitness"]
            )

            fitness_improvement = (
                (
                    cold_fitness
                    - warm_fitness
                )
                / cold_fitness
                * 100.0
            )

            # Positive = warm is better
            # Negative = cold is better

            # ------------------------------------------------
            # Runtime comparison
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
            # Winner
            # ------------------------------------------------

            if warm_fitness < cold_fitness:

                winner = "warm"

            elif cold_fitness < warm_fitness:

                winner = "cold"

            else:

                winner = "equal"

            # ------------------------------------------------
            # Warm-start validity
            # ------------------------------------------------

            warm_started = (
                warm_result[
                    "warm_started"
                ]
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            all_results.append(
                {
                    "network_seed":
                        network_seed,

                    "time_step":
                        time_step,

                    "traffic_state":
                        state,

                    "cold_iterations":
                        COLD_ITERATIONS,

                    "warm_iterations":
                        WARM_ITERATIONS,

                    "cold_fitness":
                        cold_fitness,

                    "warm_fitness":
                        warm_fitness,

                    "fitness_improvement_percent":
                        fitness_improvement,

                    "cold_runtime":
                        cold_runtime,

                    "warm_runtime":
                        warm_runtime,

                    "runtime_saving_percent":
                        runtime_saving,

                    "winner":
                        winner,

                    "warm_started":
                        warm_started,

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
                }
            )

            print(
                f"Cold={cold_fitness:.2f} "
                f"Warm={warm_fitness:.2f} "
                f"Gain={fitness_improvement:+.2f}% "
                f"Save={runtime_saving:+.2f}%"
            )

    # ========================================================
    # Save detailed CSV
    # ========================================================

    detailed_file = (
        "results/"
        "warm_start_multiseed_results.csv"
    )

    with open(
        detailed_file,
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
    # Overall statistics
    # ========================================================

    fitness_gains = [
        row[
            "fitness_improvement_percent"
        ]
        for row in all_results
    ]

    runtime_savings = [
        row[
            "runtime_saving_percent"
        ]
        for row in all_results
    ]

    warm_wins = sum(
        1
        for row in all_results
        if row["winner"] == "warm"
    )

    cold_wins = sum(
        1
        for row in all_results
        if row["winner"] == "cold"
    )

    equal = sum(
        1
        for row in all_results
        if row["winner"] == "equal"
    )

    mean_fitness_gain = statistics.mean(
        fitness_gains
    )

    std_fitness_gain = statistics.stdev(
        fitness_gains
    )

    mean_runtime_saving = statistics.mean(
        runtime_savings
    )

    std_runtime_saving = statistics.stdev(
        runtime_savings
    )

    # ========================================================
    # 95% confidence interval
    # ========================================================

    n = len(fitness_gains)

    standard_error = (
        std_fitness_gain
        / math.sqrt(n)
    )

    ci_low = (
        mean_fitness_gain
        - 1.96 * standard_error
    )

    ci_high = (
        mean_fitness_gain
        + 1.96 * standard_error
    )

    # ========================================================
    # Print overall result
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "OVERALL STATISTICAL RESULT"
    )

    print("=" * 110)

    print(
        f"\nTotal comparisons       : {n}"
    )

    print(
        f"Warm-start wins         : "
        f"{warm_wins} "
        f"({warm_wins / n * 100:.2f}%)"
    )

    print(
        f"Cold-start wins         : "
        f"{cold_wins} "
        f"({cold_wins / n * 100:.2f}%)"
    )

    print(
        f"Equal                   : "
        f"{equal}"
    )

    print(
        f"\nMean fitness improvement: "
        f"{mean_fitness_gain:+.4f}%"
    )

    print(
        f"Fitness std deviation   : "
        f"{std_fitness_gain:.4f}%"
    )

    print(
        f"95% CI                  : "
        f"[{ci_low:+.4f}%, "
        f"{ci_high:+.4f}%]"
    )

    print(
        f"\nMean runtime saving     : "
        f"{mean_runtime_saving:+.4f}%"
    )

    print(
        f"Runtime std deviation   : "
        f"{std_runtime_saving:.4f}%"
    )

    # ========================================================
    # Statistics by traffic state
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RESULT BY TRAFFIC STATE"
    )

    print("=" * 110)

    print(
        f"{'State':<18}"
        f"{'Cold Mean':>14}"
        f"{'Warm Mean':>14}"
        f"{'Fitness Gain':>15}"
        f"{'Gain Std':>12}"
        f"{'Warm Wins':>12}"
        f"{'Runtime Save':>15}"
    )

    print("-" * 110)

    state_summaries = []

    for state in TRAFFIC_STATES:

        subset = [
            row
            for row in all_results
            if row["traffic_state"]
            == state
        ]

        cold_values = [
            row["cold_fitness"]
            for row in subset
        ]

        warm_values = [
            row["warm_fitness"]
            for row in subset
        ]

        gains = [
            row[
                "fitness_improvement_percent"
            ]
            for row in subset
        ]

        savings = [
            row[
                "runtime_saving_percent"
            ]
            for row in subset
        ]

        state_cold_mean = (
            statistics.mean(
                cold_values
            )
        )

        state_warm_mean = (
            statistics.mean(
                warm_values
            )
        )

        state_gain = (
            statistics.mean(
                gains
            )
        )

        state_gain_std = (
            statistics.stdev(gains)
            if len(gains) > 1
            else 0.0
        )

        state_runtime_save = (
            statistics.mean(
                savings
            )
        )

        state_warm_wins = sum(
            1
            for row in subset
            if row["winner"] == "warm"
        )

        print(
            f"{state:<18}"
            f"{state_cold_mean:>14.3f}"
            f"{state_warm_mean:>14.3f}"
            f"{state_gain:>14.2f}%"
            f"{state_gain_std:>11.2f}%"
            f"{state_warm_wins:>12}"
            f"{state_runtime_save:>14.2f}%"
        )

        state_summaries.append(
            {
                "traffic_state":
                    state,

                "cold_mean":
                    state_cold_mean,

                "warm_mean":
                    state_warm_mean,

                "fitness_gain_percent":
                    state_gain,

                "fitness_gain_std":
                    state_gain_std,

                "warm_wins":
                    state_warm_wins,

                "comparisons":
                    len(subset),

                "runtime_saving_percent":
                    state_runtime_save,
            }
        )

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary_file = (
        "results/"
        "warm_start_multiseed_summary.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
    ) as file:

        fieldnames = list(
            state_summaries[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            state_summaries
        )

    # ========================================================
    # Final interpretation
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "INTERPRETATION"
    )

    print("=" * 110)

    if mean_fitness_gain > 0:

        print(
            "\nWarm-start produced better mean "
            "fitness than cold-start."
        )

    else:

        print(
            "\nWarm-start did not improve mean "
            "fitness overall."
        )

    if mean_runtime_saving > 0:

        print(
            "Warm-start with 30 iterations used "
            "less optimization runtime than "
            "cold-start with 50 iterations."
        )

    else:

        print(
            "Warm-start did not reduce runtime "
            "overall."
        )

    print(
        "\nDetailed results:"
    )

    print(
        f"  {detailed_file}"
    )

    print(
        "Summary results:"
    )

    print(
        f"  {summary_file}"
    )

    print(
        "\nMulti-seed benchmark completed."
    )


if __name__ == "__main__":
    main()
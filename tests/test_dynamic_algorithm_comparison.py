"""
Dynamic PSO vs QPSO vs Adaptive-QPSO

Traffic sequence:

    t0 -> normal
    t1 -> gradual 25%
    t2 -> gradual 50%
    t3 -> incident at layer 50
    t4 -> incident at layer 70
    t5 -> normal recovery

Algorithms:

    PSO
    QPSO beta=0.9
    Adaptive-QPSO beta=0.9 -> 0.5

All algorithms use:
    20 particles
    50 iterations
    same network topology
    same traffic state
    same optimizer seed
"""

import csv
import os
import random
import statistics
import time

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node

from src.traffic.traffic_model import (
    DynamicTrafficModel,
)

from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.algorithms.adaptive_qpso import AdaptiveQPSO


# ============================================================
# CONFIGURATION
# ============================================================

LAYERS = 100
BRANCHES = 3

PARTICLES = 20
ITERATIONS = 50

NETWORK_SEED = 42
OPTIMIZER_SEED = 42

FIXED_BETA = 0.9

ADAPTIVE_BETA_MAX = 0.9
ADAPTIVE_BETA_MIN = 0.5

START_NODE = "N0"
DESTINATION_NODE = f"N{LAYERS}"

RESULT_DIR = "results"

RAW_FILE = os.path.join(
    RESULT_DIR,
    "dynamic_algorithm_comparison_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "dynamic_algorithm_comparison_summary.csv",
)


# ============================================================
# BUILD NETWORK
# ============================================================

def build_network():

    rng = random.Random(
        NETWORK_SEED
    )

    graph = Graph()

    start = Node(
        START_NODE
    )

    graph.add_node(start)

    layers = []

    # --------------------------------------------------------
    # Create layers
    # --------------------------------------------------------

    for layer in range(
        1,
        LAYERS + 1,
    ):

        current_layer = []

        for branch in range(
            BRANCHES
        ):

            node = Node(
                f"L{layer}_B{branch}"
            )

            graph.add_node(node)

            current_layer.append(node)

        layers.append(
            current_layer
        )

    destination = Node(
        DESTINATION_NODE
    )

    graph.add_node(
        destination
    )

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
# APPLY TRAFFIC STATE
# ============================================================

def apply_traffic_state(
    traffic,
    graph,
    state,
):

    if state == "normal":

        traffic.normal(
            graph
        )

    elif state == "gradual_25":

        traffic.gradual_congestion(
            graph,
            level=0.25,
        )

    elif state == "gradual_50":

        traffic.gradual_congestion(
            graph,
            level=0.50,
        )

    elif state == "incident_50":

        traffic.incident(
            graph,
            layer=50,
            incident_congestion=1.0,
        )

    elif state == "incident_70":

        traffic.incident(
            graph,
            layer=70,
            incident_congestion=1.0,
        )

    else:

        raise ValueError(
            f"Unknown traffic state: {state}"
        )


# ============================================================
# ROUTE SIGNATURE
# ============================================================

def route_signature(route):

    if route is None:
        return None

    return tuple(
        node.id
        for node in route
    )


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
# EXTRACT RESULT
# ============================================================

def extract_result(
    algorithm,
    time_step,
    traffic_state,
    result,
    runtime,
    previous_route,
):

    route = result.get(
        "route"
    )

    current_route = (
        route_signature(route)
    )

    route_changed = (
        previous_route is not None
        and current_route != previous_route
    )

    history = result.get(
        "history",
        [],
    )

    initial_fitness = (
        history[0]
        if history
        else result["fitness"]
    )

    final_fitness = result[
        "fitness"
    ]

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

        "time_step": time_step,

        "traffic_state": traffic_state,

        "fitness": final_fitness,

        "initial_fitness": initial_fitness,

        "improvement_percent": improvement,

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

        "route_nodes": (
            len(route)
            if route
            else 0
        ),

        "route_changed": route_changed,

        "optimization_time": runtime,

        "route": (
            " -> ".join(
                node.id
                for node in route
            )
            if route
            else ""
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

    traffic_states = [
        "normal",
        "gradual_25",
        "gradual_50",
        "incident_50",
        "incident_70",
        "normal",
    ]

    algorithms = [
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
    ]

    total_runs = (
        len(traffic_states)
        * len(algorithms)
    )

    completed = 0

    raw_results = []

    # ========================================================
    # SEPARATE PREVIOUS ROUTE FOR EACH ALGORITHM
    # ========================================================

    previous_routes = {
        "PSO": None,
        "QPSO": None,
        "Adaptive-QPSO": None,
    }

    print()
    print("=" * 110)
    print(
        "DYNAMIC PSO vs QPSO vs ADAPTIVE-QPSO"
    )
    print("=" * 110)

    print(
        f"Network layers       : {LAYERS}"
    )

    print(
        f"Branches             : {BRANCHES}"
    )

    print(
        f"Particles            : {PARTICLES}"
    )

    print(
        f"Iterations           : {ITERATIONS}"
    )

    print(
        f"Fixed QPSO beta      : {FIXED_BETA}"
    )

    print(
        f"Adaptive beta        : "
        f"{ADAPTIVE_BETA_MAX} -> "
        f"{ADAPTIVE_BETA_MIN}"
    )

    print(
        f"Traffic states       : "
        f"{len(traffic_states)}"
    )

    print(
        f"Total optimization runs: "
        f"{total_runs}"
    )

    print()

    # ========================================================
    # SIMULATION
    # ========================================================

    for time_step, traffic_state in enumerate(
        traffic_states
    ):

        print()
        print("-" * 110)

        print(
            f"TIME STEP {time_step}: "
            f"{traffic_state}"
        )

        print("-" * 110)

        # ----------------------------------------------------
        # Each algorithm gets the same network topology and
        # same traffic state.
        # ----------------------------------------------------

        graphs = {}

        for algorithm in algorithms:

            graph = build_network()

            traffic = DynamicTrafficModel(
                seed=42,
                base_congestion=0.10,
                congestion_factor=5.0,
            )

            apply_traffic_state(
                traffic,
                graph,
                traffic_state,
            )

            graphs[
                algorithm
            ] = graph

        # ----------------------------------------------------
        # PSO
        # ----------------------------------------------------

        result, runtime = run_pso(
            graphs["PSO"],
            OPTIMIZER_SEED + time_step,
        )

        row = extract_result(
            "PSO",
            time_step,
            traffic_state,
            result,
            runtime,
            previous_routes["PSO"],
        )

        raw_results.append(row)

        previous_routes["PSO"] = (
            route_signature(
                result.get("route")
            )
        )

        completed += 1

        print(
            f"[{completed:2d}/{total_runs}] "
            f"PSO            "
            f"fitness={row['fitness']:.6f} "
            f"time={runtime:.5f}s "
            f"changed={row['route_changed']}"
        )

        # ----------------------------------------------------
        # QPSO
        # ----------------------------------------------------

        result, runtime = run_qpso(
            graphs["QPSO"],
            OPTIMIZER_SEED + time_step,
        )

        row = extract_result(
            "QPSO",
            time_step,
            traffic_state,
            result,
            runtime,
            previous_routes["QPSO"],
        )

        raw_results.append(row)

        previous_routes["QPSO"] = (
            route_signature(
                result.get("route")
            )
        )

        completed += 1

        print(
            f"[{completed:2d}/{total_runs}] "
            f"QPSO           "
            f"fitness={row['fitness']:.6f} "
            f"time={runtime:.5f}s "
            f"changed={row['route_changed']}"
        )

        # ----------------------------------------------------
        # Adaptive QPSO
        # ----------------------------------------------------

        result, runtime = (
            run_adaptive_qpso(
                graphs["Adaptive-QPSO"],
                OPTIMIZER_SEED + time_step,
            )
        )

        row = extract_result(
            "Adaptive-QPSO",
            time_step,
            traffic_state,
            result,
            runtime,
            previous_routes[
                "Adaptive-QPSO"
            ],
        )

        raw_results.append(row)

        previous_routes[
            "Adaptive-QPSO"
        ] = route_signature(
            result.get("route")
        )

        completed += 1

        print(
            f"[{completed:2d}/{total_runs}] "
            f"Adaptive-QPSO  "
            f"fitness={row['fitness']:.6f} "
            f"time={runtime:.5f}s "
            f"changed={row['route_changed']}"
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
    # SUMMARY BY TRAFFIC STATE
    # ========================================================

    summary = []

    for time_step, traffic_state in enumerate(
        traffic_states
    ):

        for algorithm in algorithms:

            rows = [
                row
                for row in raw_results
                if (
                    row["time_step"]
                    == time_step
                    and row["algorithm"]
                    == algorithm
                )
            ]

            fitness_values = [
                row["fitness"]
                for row in rows
            ]

            runtime_values = [
                row["optimization_time"]
                for row in rows
            ]

            congestion_values = [
                row["congestion"]
                for row in rows
            ]

            summary.append(
                {
                    "time_step": time_step,

                    "traffic_state": traffic_state,

                    "algorithm": algorithm,

                    "fitness": fitness_values[0],

                    "travel_time": rows[0][
                        "travel_time"
                    ],

                    "distance": rows[0][
                        "distance"
                    ],

                    "congestion": (
                        congestion_values[0]
                    ),

                    "optimization_time": (
                        runtime_values[0]
                    ),

                    "route_changed": (
                        rows[0][
                            "route_changed"
                        ]
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
    # PRINT STATE-BY-STATE COMPARISON
    # ========================================================

    print()
    print("=" * 125)
    print(
        "DYNAMIC ALGORITHM COMPARISON SUMMARY"
    )
    print("=" * 125)

    print(
        f"{'Time':<7}"
        f"{'Traffic':<16}"
        f"{'Algorithm':<20}"
        f"{'Fitness':<15}"
        f"{'Travel Time':<16}"
        f"{'Congestion':<13}"
        f"{'Runtime':<12}"
        f"{'Changed':<10}"
    )

    print("-" * 125)

    for row in summary:

        print(
            f"{row['time_step']:<7}"
            f"{row['traffic_state']:<16}"
            f"{row['algorithm']:<20}"
            f"{row['fitness']:<15.6f}"
            f"{row['travel_time']:<16.6f}"
            f"{row['congestion']:<13.6f}"
            f"{row['optimization_time']:<12.6f}"
            f"{str(row['route_changed']):<10}"
        )

    # ========================================================
    # WIN COUNT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "DYNAMIC SCENARIO WIN COUNT"
    )
    print("=" * 80)

    win_counts = {
        "PSO": 0,
        "QPSO": 0,
        "Adaptive-QPSO": 0,
    }

    for time_step, traffic_state in enumerate(
        traffic_states
    ):

        rows = [
            row
            for row in summary
            if row["time_step"] == time_step
        ]

        winner = min(
            rows,
            key=lambda row:
                row["fitness"],
        )

        win_counts[
            winner["algorithm"]
        ] += 1

        print(
            f"Time {time_step} "
            f"{traffic_state:<15}"
            f"winner = "
            f"{winner['algorithm']:<18}"
            f"fitness = "
            f"{winner['fitness']:.6f}"
        )

    print()

    for algorithm in [
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
    ]:

        print(
            f"{algorithm:<20}: "
            f"{win_counts[algorithm]}"
        )

    # ========================================================
    # ROUTE CHANGE COUNT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "ROUTE ADAPTATION COUNT"
    )
    print("=" * 80)

    for algorithm in [
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
    ]:

        rows = [
            row
            for row in summary
            if row["algorithm"]
            == algorithm
        ]

        changes = sum(
            1
            for row in rows
            if row["route_changed"]
        )

        print(
            f"{algorithm:<20}: "
            f"{changes} route changes"
        )

    # ========================================================
    # AVERAGE RUNTIME
    # ========================================================

    print()
    print("=" * 80)
    print(
        "AVERAGE OPTIMIZATION TIME"
    )
    print("=" * 80)

    for algorithm in [
        "PSO",
        "QPSO",
        "Adaptive-QPSO",
    ]:

        values = [
            row["optimization_time"]
            for row in summary
            if row["algorithm"]
            == algorithm
        ]

        print(
            f"{algorithm:<20}: "
            f"{statistics.mean(values):.6f} s"
        )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FILES GENERATED"
    )
    print("=" * 80)

    print(
        RAW_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "Dynamic algorithm comparison "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
"""
Dynamic Traffic Routing Experiment

Tests whether Adaptive-QPSO can react to changing
traffic conditions and select new routes.

Traffic sequence:

    0 -> normal
    1 -> gradual congestion
    2 -> incident at layer 50
    3 -> incident moves to layer 70
    4 -> normal recovery

The same network is used throughout the simulation.
"""

import csv
import os
import time

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node

from src.traffic.traffic_model import (
    DynamicTrafficModel,
)

from src.algorithms.adaptive_qpso import (
    AdaptiveQPSO,
)


# ============================================================
# CONFIGURATION
# ============================================================

LAYERS = 100
BRANCHES = 3

PARTICLES = 20
ITERATIONS = 50

NETWORK_SEED = 42
OPTIMIZER_SEED = 42

START_NODE = "N0"
DESTINATION_NODE = f"N{LAYERS}"

RESULT_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "dynamic_routing_results.csv",
)


# ============================================================
# BUILD NETWORK
# ============================================================

def build_network():

    import random

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
# ROUTE SIGNATURE
# ============================================================

def route_signature(route):

    if not route:
        return None

    return tuple(
        node.id
        for node in route
    )


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
            f"Unknown traffic state: "
            f"{state}"
        )


# ============================================================
# RUN OPTIMIZATION
# ============================================================

def optimize_route(
    graph,
    seed,
):

    start_time = time.perf_counter()

    optimizer = AdaptiveQPSO(
        graph=graph,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta_max=0.9,
        beta_min=0.5,
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
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # One persistent network
    # --------------------------------------------------------

    graph = build_network()

    traffic = DynamicTrafficModel(
        seed=42,
        base_congestion=0.10,
        congestion_factor=5.0,
    )

    # --------------------------------------------------------
    # Dynamic traffic sequence
    # --------------------------------------------------------

    traffic_states = [
        "normal",
        "gradual_25",
        "gradual_50",
        "incident_50",
        "incident_70",
        "normal",
    ]

    results = []

    previous_route = None

    print()
    print("=" * 100)
    print(
        "DYNAMIC TRAFFIC ROUTING EXPERIMENT"
    )
    print("=" * 100)

    print(
        f"Network layers : {LAYERS}"
    )

    print(
        f"Branches       : {BRANCHES}"
    )

    print(
        f"Particles      : {PARTICLES}"
    )

    print(
        f"Iterations     : {ITERATIONS}"
    )

    print()

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for time_step, state in enumerate(
        traffic_states
    ):

        # ----------------------------------------------------
        # Update traffic
        # ----------------------------------------------------

        apply_traffic_state(
            traffic,
            graph,
            state,
        )

        # ----------------------------------------------------
        # Optimize route
        # ----------------------------------------------------

        result, runtime = (
            optimize_route(
                graph,
                OPTIMIZER_SEED
                + time_step,
            )
        )

        route = result.get(
            "route"
        )

        route_ids = [
            node.id
            for node in route
        ]

        # ----------------------------------------------------
        # Detect route change
        # ----------------------------------------------------

        current_signature = (
            route_signature(route)
        )

        route_changed = (
            previous_route is not None
            and current_signature
            != previous_route
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        row = {
            "time_step": time_step,

            "traffic_state": state,

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

            "route_nodes": len(
                route
            ),

            "route_changed": (
                route_changed
            ),

            "optimization_time": (
                runtime
            ),

            "route": " -> ".join(
                route_ids
            ),
        }

        results.append(
            row
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Time {time_step}: "
            f"{state}"
        )

        print(
            f"  Fitness       : "
            f"{row['fitness']:.6f}"
        )

        print(
            f"  Travel time   : "
            f"{row['travel_time']:.6f}"
        )

        print(
            f"  Distance      : "
            f"{row['distance']:.6f}"
        )

        print(
            f"  Congestion    : "
            f"{row['congestion']:.6f}"
        )

        print(
            f"  Route changed : "
            f"{row['route_changed']}"
        )

        print(
            f"  Optimization  : "
            f"{row['optimization_time']:.6f} s"
        )

        print(
            f"  Route length  : "
            f"{row['route_nodes']}"
        )

        print()

        previous_route = (
            current_signature
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    changed_routes = sum(
        1
        for row in results
        if row["route_changed"]
    )

    average_runtime = (
        sum(
            row["optimization_time"]
            for row in results
        )
        / len(results)
    )

    initial_fitness = results[0][
        "fitness"
    ]

    final_fitness = results[-1][
        "fitness"
    ]

    print("=" * 100)
    print(
        "DYNAMIC ROUTING SUMMARY"
    )
    print("=" * 100)

    print(
        f"Simulation steps     : "
        f"{len(results)}"
    )

    print(
        f"Route changes        : "
        f"{changed_routes}"
    )

    print(
        f"Average optimization : "
        f"{average_runtime:.6f} s"
    )

    print(
        f"Initial fitness      : "
        f"{initial_fitness:.6f}"
    )

    print(
        f"Final fitness        : "
        f"{final_fitness:.6f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Dynamic routing experiment "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
import time

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node

from src.optimization.fitness import FitnessFunction

from src.traffic.scenario import (
    TrafficScenario,
    create_scenario,
)

from src.traffic.traffic_model import (
    TrafficModel,
)


def build_graph():

    nodes = [
        Node("N0"),
        Node("N1"),
        Node("N2"),
        Node("N3"),
        Node("N4"),
        Node("N5"),
    ]

    graph = Graph()

    n0, n1, n2, n3, n4, n5 = nodes

    edges = [

        # Main route
        Edge(
            n0,
            n2,
            distance=8,
            travel_time=9,
            congestion=0.10,
        ),

        Edge(
            n2,
            n3,
            distance=7,
            travel_time=8,
            congestion=0.12,
        ),

        Edge(
            n3,
            n5,
            distance=7,
            travel_time=7.84,
            congestion=0.15,
        ),

        # Alternative route 1
        Edge(
            n0,
            n1,
            distance=12,
            travel_time=13,
            congestion=0.25,
        ),

        Edge(
            n1,
            n3,
            distance=10,
            travel_time=12,
            congestion=0.20,
        ),

        # Alternative route 2
        Edge(
            n2,
            n4,
            distance=9,
            travel_time=10,
            congestion=0.18,
        ),

        Edge(
            n4,
            n5,
            distance=8,
            travel_time=9,
            congestion=0.20,
        ),

        # Additional alternative
        Edge(
            n1,
            n4,
            distance=11,
            travel_time=13,
            congestion=0.25,
        ),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


def route_string(route):

    if route is None:
        return "NO ROUTE"

    return " -> ".join(
        node.id
        for node in route
    )


def run_algorithm(
    algorithm,
    start,
    destination,
):

    start_time = time.perf_counter()

    result = algorithm.optimize(
        start,
        destination,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return result, runtime


def run_scenario(
    scenario_name,
    graph,
):

    fitness_function = FitnessFunction()

    traffic_model = TrafficModel(
        graph
    )

    # ---------------------------------------------------------
    # Apply scenario to a COPY of the graph.
    # ---------------------------------------------------------

    if scenario_name == "normal":

        scenario = create_scenario(
            TrafficScenario.NORMAL
        )

    elif scenario_name == "peak":

        scenario = create_scenario(
            TrafficScenario.PEAK
        )

    elif scenario_name == "congested":

        scenario = create_scenario(
            TrafficScenario.CONGESTED
        )

    elif scenario_name == "incident":

        scenario = traffic_model.incident_scenario(
            "N2",
            "N3",
            travel_time_multiplier=3.0,
            congestion=0.95,
        )

    elif scenario_name == "blockage":

        scenario = traffic_model.road_blockage_scenario(
            "N2",
            "N3",
        )

    else:

        raise ValueError(
            f"Unknown scenario: {scenario_name}"
        )

    scenario_graph = (
        traffic_model.apply_scenario(
            scenario
        )
    )

    start = "N0"
    destination = "N5"

    # =========================================================
    # DIJKSTRA
    # =========================================================

    dijkstra = Dijkstra(
        scenario_graph,
        fitness_function=fitness_function,
    )

    dijkstra_result, dijkstra_runtime = (
        run_algorithm(
            dijkstra,
            start,
            destination,
        )
    )

    # =========================================================
    # PSO
    # =========================================================

    pso = PSO(
        scenario_graph,
        particles=30,
        iterations=100,
        seed=42,
        fitness_function=fitness_function,
    )

    pso_result, pso_runtime = (
        run_algorithm(
            pso,
            start,
            destination,
        )
    )

    # =========================================================
    # QPSO
    # =========================================================

    qpso = QPSO(
        scenario_graph,
        particles=30,
        iterations=100,
        seed=42,
        fitness_function=fitness_function,
    )

    qpso_result, qpso_runtime = (
        run_algorithm(
            qpso,
            start,
            destination,
        )
    )

    # =========================================================
    # DISPLAY
    # =========================================================

    print()
    print("=" * 70)
    print(
        f"TRAFFIC SCENARIO: "
        f"{scenario_name.upper()}"
    )
    print("=" * 70)

    print()
    print("Dijkstra")
    print("-" * 30)

    print(
        "Route:",
        route_string(
            dijkstra_result["route"]
        ),
    )

    print(
        "Fitness:",
        f"{dijkstra_result['fitness']:.4f}",
    )

    print(
        "Travel time:",
        f"{dijkstra_result['travel_time']:.4f}",
    )

    print(
        "Distance:",
        f"{dijkstra_result['distance']:.4f}",
    )

    print(
        "Congestion:",
        f"{dijkstra_result['congestion']:.4f}",
    )

    print(
        "Runtime:",
        f"{dijkstra_runtime:.6f}s",
    )

    print()
    print("PSO")
    print("-" * 30)

    print(
        "Route:",
        route_string(
            pso_result["route"]
        ),
    )

    print(
        "Fitness:",
        f"{pso_result['fitness']:.4f}",
    )

    print(
        "Travel time:",
        f"{pso_result['travel_time']:.4f}",
    )

    print(
        "Distance:",
        f"{pso_result['distance']:.4f}",
    )

    print(
        "Congestion:",
        f"{pso_result['congestion']:.4f}",
    )

    print(
        "Runtime:",
        f"{pso_runtime:.6f}s",
    )

    print()
    print("QPSO")
    print("-" * 30)

    print(
        "Route:",
        route_string(
            qpso_result["route"]
        ),
    )

    print(
        "Fitness:",
        f"{qpso_result['fitness']:.4f}",
    )

    print(
        "Travel time:",
        f"{qpso_result['travel_time']:.4f}",
    )

    print(
        "Distance:",
        f"{qpso_result['distance']:.4f}",
    )

    print(
        "Congestion:",
        f"{qpso_result['congestion']:.4f}",
    )

    print(
        "Runtime:",
        f"{qpso_runtime:.6f}s",
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    results = {
        "Dijkstra": dijkstra_result,
        "PSO": pso_result,
        "QPSO": qpso_result,
    }

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    for name, result in results.items():

        print(
            f"{name:<10} "
            f"fitness={result['fitness']:.4f} "
            f"time={result['travel_time']:.4f} "
            f"distance={result['distance']:.4f} "
            f"congestion={result['congestion']:.4f}"
        )

    best_algorithm = min(
        results,
        key=lambda name:
        results[name]["fitness"],
    )

    print()
    print(
        "Best fitness:",
        best_algorithm,
    )

    return {
        "scenario": scenario_name,
        "graph": scenario_graph,
        "dijkstra": (
            dijkstra_result,
            dijkstra_runtime,
        ),
        "pso": (
            pso_result,
            pso_runtime,
        ),
        "qpso": (
            qpso_result,
            qpso_runtime,
        ),
    }


def test_dynamic_benchmark():

    graph = build_graph()

    scenarios = [
        "normal",
        "peak",
        "congested",
        "incident",
        "blockage",
    ]

    results = []

    for scenario_name in scenarios:

        result = run_scenario(
            scenario_name,
            graph,
        )

        results.append(
            result
        )

        assert (
            result["dijkstra"][0]["route"]
            is not None
        )

        assert (
            result["pso"][0]["route"]
            is not None
        )

        assert (
            result["qpso"][0]["route"]
            is not None
        )

    assert len(results) == 5
# dashboard/optimizer_service.py

import sys
import copy
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data.scenario_graph_generator import (
    ScenarioGraphGenerator,
)

from src.data.graph_generator import (
    LargeGraphGenerator,
)

from src.domain.customer import Customer
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem
from src.domain.edge import Edge

from src.optimization.config import (
    OptimizationConfig,
)

from src.optimization.vrp_fitness import (
    VRPFitnessFunction,
)

from src.algorithms.vrp_pso import (
    VRPPSO,
)

from src.algorithms.vrp_qpso import (
    VRPQPSO,
)

from src.algorithms.warm_vrp_qpso import (
    WarmVRPQPSO,
)

from src.algorithms.decomposed_warm_qpso import (
    DecomposedWarmQPSO,
)

from src.traffic.simulated_data import (
    SimulatedTrafficData,
)


# ============================================================
# TRAFFIC DATA
# ============================================================

TRAFFIC_DATA = SimulatedTrafficData()


SIMULATED_SCENARIOS = {
    "normal",
    "morning peak",
    "heavy congestion",
    "evening peak",
    "accident",
    "road closure",
}


CONTROLLED_SCENARIOS = {
    "upper route incident",
    "upper route blockage",
    "lower route incident",
}


# ============================================================
# AVAILABLE NETWORK SIZES
# ============================================================

def available_network_sizes():
    """
    Network sizes supported by the dashboard.

    6:
        Controlled demonstration network.

    50:
        Medium network.

    100:
        Recommended SIH demonstration.

    250:
        Large network.

    500:
        Scalability demonstration.
    """

    return [
        6,
        50,
        100,
        250,
        500,
    ]


# ============================================================
# AVAILABLE ALGORITHMS
# ============================================================

def available_algorithms():

    return [
        "PSO",
        "QPSO",
        "Warm-QPSO",
        "Decomposed Warm-QPSO",
    ]


# ============================================================
# AVAILABLE SCENARIOS
# ============================================================

def available_scenarios():

    return [
        "Normal",
        "Morning Peak",
        "Heavy Congestion",
        "Evening Peak",
        "Accident",
        "Road Closure",
    ]


# ============================================================
# BASE GRAPH
# ============================================================

def get_base_graph(
    scenario="Normal",
    network_size=6,
    edges_per_node=3,
    seed=42,
):
    """
    Generate the transportation network.

    Parameters
    ----------
    scenario:
        Traffic or controlled scenario.

    network_size:
        Number of network nodes.

    edges_per_node:
        Number of randomly generated outgoing edges
        per node for large networks.

    seed:
        Random seed for deterministic graph generation.

    Returns
    -------
    Graph
    """

    scenario_key = str(
        scenario
    ).strip().lower()

    network_size = int(
        network_size
    )

    edges_per_node = int(
        edges_per_node
    )

    seed = int(seed)

    if network_size < 6:

        raise ValueError(
            "network_size must be at least 6."
        )

    if edges_per_node < 1:

        raise ValueError(
            "edges_per_node must be at least 1."
        )

    # ========================================================
    # CONTROLLED 6-NODE NETWORK
    # ========================================================

    if network_size == 6:

        generator = (
            ScenarioGraphGenerator()
        )

        if scenario_key == "normal":

            return generator.generate()

        if (
            scenario_key
            == "upper route incident"
        ):

            return (
                generator
                .upper_route_incident()
            )

        if (
            scenario_key
            == "upper route blockage"
        ):

            return (
                generator
                .upper_route_blockage()
            )

        if (
            scenario_key
            == "lower route incident"
        ):

            return (
                generator
                .lower_route_incident()
            )

        if (
            scenario_key
            in SIMULATED_SCENARIOS
        ):

            return generator.generate()

        raise ValueError(
            f"Unknown traffic scenario: "
            f"{scenario}"
        )

    # ========================================================
    # LARGE SYNTHETIC NETWORK
    # ========================================================

    generator = LargeGraphGenerator(
        num_nodes=network_size,
        edges_per_node=edges_per_node,
        seed=seed,
    )

    return generator.generate()


# ============================================================
# BIDIRECTIONAL GRAPH
# ============================================================

def make_bidirectional_graph(graph):
    """
    Add reverse edges to a graph.

    Existing reverse edges are not duplicated.
    """

    new_graph = copy.deepcopy(
        graph
    )

    existing_edges = set()

    for edge in new_graph.get_edges():

        existing_edges.add(
            (
                str(edge.source.id),
                str(edge.target.id),
            )
        )

    reverse_edges = []

    for edge in new_graph.get_edges():

        reverse_key = (
            str(edge.target.id),
            str(edge.source.id),
        )

        if reverse_key in existing_edges:

            continue

        reverse_edges.append(
            Edge(
                edge.target,
                edge.source,
                distance=edge.distance,
                travel_time=edge.travel_time,
                congestion=edge.congestion,
                blocked=edge.blocked,
            )
        )

    for edge in reverse_edges:

        new_graph.add_edge(
            edge
        )

    return new_graph


# ============================================================
# APPLY SIMULATED TRAFFIC
# ============================================================

def apply_simulated_traffic(
    graph,
    scenario,
):
    """
    Apply traffic data from the simulated CSV dataset.

    For the six-node controlled network, the CSV contains
    explicit traffic conditions for E1-E6.

    For larger synthetic networks, CSV traffic is applied
    wherever a matching source-target edge exists.
    """

    records = TRAFFIC_DATA.get(
        scenario
    )

    if not records:

        return copy.deepcopy(
            graph
        )

    updated_graph = copy.deepcopy(
        graph
    )

    # --------------------------------------------------------
    # Edge lookup
    # --------------------------------------------------------

    edge_map = {}

    for edge in updated_graph.get_edges():

        key = (
            str(edge.source.id),
            str(edge.target.id),
        )

        edge_map[key] = edge

    # --------------------------------------------------------
    # Apply traffic
    # --------------------------------------------------------

    for record in records:

        source = str(
            record["source"]
        )

        target = str(
            record["target"]
        )

        key = (
            source,
            target,
        )

        edge = edge_map.get(
            key
        )

        if edge is None:

            continue

        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        edge.distance = float(
            record["distance_km"]
        )

        # ----------------------------------------------------
        # Congestion
        # ----------------------------------------------------

        edge.congestion = float(
            record["congestion"]
        )

        # ----------------------------------------------------
        # Blocked
        # ----------------------------------------------------

        edge.blocked = bool(
            record["blocked"]
        )

        # ----------------------------------------------------
        # Traffic speed
        # ----------------------------------------------------

        traffic_speed = float(
            record[
                "traffic_speed_kmph"
            ]
        )

        # ----------------------------------------------------
        # Travel time
        # ----------------------------------------------------

        if edge.blocked:

            edge.travel_time = float(
                "inf"
            )

        elif traffic_speed > 0:

            edge.travel_time = (
                edge.distance
                / traffic_speed
                * 60.0
            )

        else:

            edge.travel_time = (
                edge.distance
                * 60.0
            )

    return updated_graph


# ============================================================
# GET GRAPH
# ============================================================

def get_graph(
    scenario="Normal",
    network_size=6,
    edges_per_node=3,
    seed=42,
):
    """
    Complete graph pipeline:

        Scenario
            ↓
        Base graph
            ↓
        Simulated traffic
            ↓
        Bidirectional graph
    """

    scenario = str(
        scenario
    ).strip()

    graph = get_base_graph(
        scenario=scenario,
        network_size=network_size,
        edges_per_node=edges_per_node,
        seed=seed,
    )

    # --------------------------------------------------------
    # Apply CSV traffic
    # --------------------------------------------------------

    if (
        scenario.lower()
        in SIMULATED_SCENARIOS
    ):

        graph = apply_simulated_traffic(
            graph,
            scenario,
        )

    # --------------------------------------------------------
    # Make bidirectional
    # --------------------------------------------------------

    graph = make_bidirectional_graph(
        graph
    )

    return graph


# ============================================================
# BUILD VRP PROBLEM
# ============================================================

def build_problem(
    graph,
    number_of_customers=20,
    number_of_vehicles=2,
):
    """
    Build a VRP problem.

    The first graph node is the depot.

    Customer locations are assigned to unique network
    nodes whenever enough network nodes are available.

    If there are more customers than physical nodes,
    locations are reused cyclically.
    """

    number_of_customers = int(
        number_of_customers
    )

    number_of_vehicles = int(
        number_of_vehicles
    )

    if number_of_customers < 1:

        raise ValueError(
            "number_of_customers must be at least 1."
        )

    if number_of_vehicles < 1:

        raise ValueError(
            "number_of_vehicles must be at least 1."
        )

    # ========================================================
    # NETWORK NODES
    # ========================================================

    nodes = list(
        graph._node_map.values()
    )

    if len(nodes) < 2:

        raise ValueError(
            "Graph must contain at least two nodes."
        )

    # --------------------------------------------------------
    # Depot
    # --------------------------------------------------------

    depot = str(
        nodes[0].id
    )

    # --------------------------------------------------------
    # Customer nodes
    # --------------------------------------------------------

    customer_nodes = [
        node
        for node in nodes
        if str(node.id) != depot
    ]

    if not customer_nodes:

        raise ValueError(
            "No customer nodes available."
        )

    # ========================================================
    # CREATE CUSTOMERS
    # ========================================================

    customers = []

    for i in range(
        number_of_customers
    ):

        node = customer_nodes[
            i % len(customer_nodes)
        ]

        customers.append(
            Customer(
                id=f"C{i + 1}",
                demand=1.0,
                node_id=str(
                    node.id
                ),
            )
        )

    # ========================================================
    # VEHICLE CAPACITY
    # ========================================================

    capacity_per_vehicle = max(
        1.0,
        (
            number_of_customers
            / number_of_vehicles
        )
        * 1.25,
    )

    # ========================================================
    # CREATE VEHICLES
    # ========================================================

    vehicles = []

    for i in range(
        number_of_vehicles
    ):

        vehicles.append(
            Vehicle(
                id=f"V{i + 1}",
                capacity=capacity_per_vehicle,
                start_node=depot,
                end_node=depot,
            )
        )

    # ========================================================
    # CREATE VRP
    # ========================================================

    return VRPProblem(
        graph=graph,
        depot=depot,
        customers=customers,
        vehicles=vehicles,
    )


# ============================================================
# CREATE OPTIMIZER
# ============================================================

def create_optimizer(
    algorithm_name,
    problem,
    config=None,
):
    """
    Create the requested optimization algorithm.
    """

    if config is None:

        config = OptimizationConfig()

    fitness_function = (
        VRPFitnessFunction()
    )

    # ========================================================
    # PSO
    # ========================================================

    if algorithm_name == "PSO":

        return VRPPSO(
            problem=problem,
            particles=config.particles,
            iterations=config.iterations,
            inertia=config.inertia,
            cognitive=config.cognitive,
            social=config.social,
            seed=config.seed,
            fitness_function=fitness_function,
        )

    # ========================================================
    # QPSO
    # ========================================================

    if algorithm_name == "QPSO":

        return VRPQPSO(
            problem=problem,
            particles=config.particles,
            iterations=config.iterations,
            beta=config.beta,
            seed=config.seed,
            fitness_function=fitness_function,
        )

    # ========================================================
    # WARM-QPSO
    # ========================================================

    if algorithm_name == "Warm-QPSO":

        return WarmVRPQPSO(
            problem=problem,
            particles=config.particles,
            iterations=config.iterations,
            beta=config.beta,
            seed=config.seed,
            fitness_function=fitness_function,
            warm_iterations=config.warm_iterations,
            warm_particles=config.warm_particles,
        )

    # ========================================================
    # DECOMPOSED WARM-QPSO
    # ========================================================

    if algorithm_name == "Decomposed Warm-QPSO":

        effective_clusters = min(
            config.clusters,
            problem.number_of_customers,
            problem.number_of_vehicles,
        )

        effective_clusters = max(
            1,
            effective_clusters,
        )

        return DecomposedWarmQPSO(
            problem=problem,
            clusters=effective_clusters,
            particles=config.particles,
            iterations=config.iterations,
            beta=config.beta,
            seed=config.seed,
            fitness_function=fitness_function,
            warm_iterations=config.warm_iterations,
            warm_particles=config.warm_particles,
            repair=config.repair_enabled,
            repair_rounds=config.repair_rounds,
            max_boundary_customers=(
                config.max_boundary_customers
            ),
            max_target_vehicles=(
                config.max_target_vehicles
            ),
            try_swaps=config.try_swaps,
        )

    raise ValueError(
        f"Unknown algorithm: "
        f"{algorithm_name}"
    )


# ============================================================
# NORMALIZE VALIDITY
# ============================================================

def normalize_result_validity(
    result
):
    """
    Ensure the optimizer result contains a top-level
    'valid' field.

    Older algorithm implementations expose validity
    inside vehicle_results.
    """

    if "valid" in result:

        return result

    vehicle_results = result.get(
        "vehicle_results",
        {},
    )

    if isinstance(
        vehicle_results,
        dict,
    ):

        if not vehicle_results:

            result["valid"] = (
                result.get(
                    "solution"
                )
                is not None
            )

        else:

            result["valid"] = all(
                vehicle_result.get(
                    "valid",
                    False,
                )
                for vehicle_result
                in vehicle_results.values()
            )

    elif isinstance(
        vehicle_results,
        list,
    ):

        result["valid"] = all(
            vehicle_result.get(
                "valid",
                False,
            )
            for vehicle_result
            in vehicle_results
        )

    else:

        result["valid"] = (
            result.get(
                "solution"
            )
            is not None
        )

    return result


# ============================================================
# SERIALIZE GRAPH
# ============================================================

def serialize_graph(
    graph
):
    """
    Convert graph objects into Plotly-friendly dictionaries.
    """

    graph_nodes = []

    for node in graph._node_map.values():

        graph_nodes.append(
            str(node.id)
        )

    graph_edges = []

    for edge in graph.get_edges():

        travel_time = (
            edge.travel_time
        )

        if travel_time == float(
            "inf"
        ):

            travel_time = 999999.0

        graph_edges.append(
            {
                "source": str(
                    edge.source.id
                ),
                "target": str(
                    edge.target.id
                ),
                "distance": float(
                    edge.distance
                ),
                "travel_time": float(
                    travel_time
                ),
                "congestion": float(
                    edge.congestion
                ),
                "blocked": bool(
                    edge.blocked
                ),
            }
        )

    return (
        graph_nodes,
        graph_edges,
    )


# ============================================================
# RUN OPTIMIZER
# ============================================================

def run_optimizer(
    scenario="Normal",
    algorithm_name="Decomposed Warm-QPSO",
    customers=20,
    vehicles=2,

    # --------------------------------------------------------
    # Large-network settings
    # --------------------------------------------------------

    network_size=100,
    edges_per_node=3,

    # --------------------------------------------------------
    # Configuration object
    # --------------------------------------------------------

    config=None,

    # --------------------------------------------------------
    # Dashboard-compatible configuration
    # --------------------------------------------------------

    particles=None,
    iterations=None,
    inertia=None,
    cognitive=None,
    social=None,
    beta=None,
    warm_iterations=None,
    warm_particles=None,
    clusters=None,
    repair_enabled=None,
    repair_rounds=None,
    max_boundary_customers=None,
    max_target_vehicles=None,
    try_swaps=None,
    seed=None,
):
    """
    Main Q-ROUTE optimization service.

    Supports:

        run_optimizer(
            scenario="Normal",
            algorithm_name="PSO",
            customers=20,
            vehicles=2,
            network_size=100,
        )

    and dashboard-style overrides:

        particles=10
        iterations=20
        beta=0.7
        clusters=5
    """

    # ========================================================
    # CONFIG
    # ========================================================

    if config is None:

        config = OptimizationConfig()

    # ========================================================
    # APPLY OVERRIDES
    # ========================================================

    if particles is not None:

        config.particles = int(
            particles
        )

    if iterations is not None:

        config.iterations = int(
            iterations
        )

    if inertia is not None:

        config.inertia = float(
            inertia
        )

    if cognitive is not None:

        config.cognitive = float(
            cognitive
        )

    if social is not None:

        config.social = float(
            social
        )

    if beta is not None:

        config.beta = float(
            beta
        )

    if warm_iterations is not None:

        config.warm_iterations = int(
            warm_iterations
        )

    if warm_particles is not None:

        config.warm_particles = int(
            warm_particles
        )

    if clusters is not None:

        config.clusters = int(
            clusters
        )

    if repair_enabled is not None:

        config.repair_enabled = bool(
            repair_enabled
        )

    if repair_rounds is not None:

        config.repair_rounds = int(
            repair_rounds
        )

    if max_boundary_customers is not None:

        config.max_boundary_customers = int(
            max_boundary_customers
        )

    if max_target_vehicles is not None:

        config.max_target_vehicles = int(
            max_target_vehicles
        )

    if try_swaps is not None:

        config.try_swaps = bool(
            try_swaps
        )

    if seed is not None:

        config.seed = int(
            seed
        )

    # ========================================================
    # VALIDATE CONFIG
    # ========================================================

    config.__post_init__()

    # ========================================================
    # VALIDATE ALGORITHM
    # ========================================================

    if (
        algorithm_name
        not in available_algorithms()
    ):

        raise ValueError(
            f"Unknown algorithm: "
            f"{algorithm_name}"
        )

    # ========================================================
    # VALIDATE NETWORK
    # ========================================================

    network_size = int(
        network_size
    )

    edges_per_node = int(
        edges_per_node
    )

    if network_size < 6:

        raise ValueError(
            "network_size must be at least 6."
        )

    if edges_per_node < 1:

        raise ValueError(
            "edges_per_node must be at least 1."
        )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    graph = get_graph(
        scenario=scenario,
        network_size=network_size,
        edges_per_node=edges_per_node,
        seed=config.seed,
    )

    # ========================================================
    # BUILD VRP
    # ========================================================

    problem = build_problem(
        graph,
        number_of_customers=customers,
        number_of_vehicles=vehicles,
    )

    # ========================================================
    # CREATE OPTIMIZER
    # ========================================================

    optimizer = create_optimizer(
        algorithm_name,
        problem,
        config,
    )

    # ========================================================
    # OPTIMIZE
    # ========================================================

    start_time = time.perf_counter()

    result = optimizer.optimize()

    measured_runtime = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # NORMALIZE VALIDITY
    # ========================================================

    result = normalize_result_validity(
        result
    )

    # ========================================================
    # EXECUTION TIME
    # ========================================================

    if (
        result.get(
            "execution_time"
        )
        is None
    ):

        result["execution_time"] = (
            measured_runtime
        )

    # ========================================================
    # METADATA
    # ========================================================

    result["scenario"] = (
        scenario
    )

    result["algorithm"] = (
        algorithm_name
    )

    result["customers"] = int(
        customers
    )

    result["vehicles"] = int(
        vehicles
    )

    result["network_size"] = (
        network_size
    )

    result["edges_per_node"] = (
        edges_per_node
    )

    result["configuration"] = (
        config.to_dict()
    )

    result["traffic_data_source"] = (
        "Simulated traffic dataset"
    )

    # ========================================================
    # EFFECTIVE CLUSTERS
    # ========================================================

    if (
        algorithm_name
        == "Decomposed Warm-QPSO"
    ):

        result[
            "effective_clusters"
        ] = min(
            config.clusters,
            int(customers),
            int(vehicles),
        )

    else:

        result[
            "effective_clusters"
        ] = None

    # ========================================================
    # GRAPH SERIALIZATION
    # ========================================================

    (
        graph_nodes,
        graph_edges,
    ) = serialize_graph(
        graph
    )

    result["graph_nodes"] = (
        graph_nodes
    )

    result["graph_edges"] = (
        graph_edges
    )

    # ========================================================
    # TRAFFIC SUMMARY
    # ========================================================

    result["traffic_summary"] = (
        get_traffic_summary(
            scenario
        )
    )

    # ========================================================
    # TRAFFIC RECORDS
    # ========================================================

    result["traffic_records"] = (
        get_traffic_records(
            scenario
        )
    )

    # ========================================================
    # RETURN
    # ========================================================

    return result


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

def run_algorithm_comparison(
    scenario="Normal",
    customers=20,
    vehicles=2,
    network_size=100,
    edges_per_node=3,
    config=None,

    particles=None,
    iterations=None,
    inertia=None,
    cognitive=None,
    social=None,
    beta=None,
    warm_iterations=None,
    warm_particles=None,
    clusters=None,
    repair_enabled=None,
    repair_rounds=None,
    max_boundary_customers=None,
    max_target_vehicles=None,
    try_swaps=None,
    seed=None,
):
    """
    Run all four algorithms using the same configuration.
    """

    if config is None:

        config = OptimizationConfig()

    # --------------------------------------------------------
    # Apply configuration overrides
    # --------------------------------------------------------

    if particles is not None:

        config.particles = int(
            particles
        )

    if iterations is not None:

        config.iterations = int(
            iterations
        )

    if inertia is not None:

        config.inertia = float(
            inertia
        )

    if cognitive is not None:

        config.cognitive = float(
            cognitive
        )

    if social is not None:

        config.social = float(
            social
        )

    if beta is not None:

        config.beta = float(
            beta
        )

    if warm_iterations is not None:

        config.warm_iterations = int(
            warm_iterations
        )

    if warm_particles is not None:

        config.warm_particles = int(
            warm_particles
        )

    if clusters is not None:

        config.clusters = int(
            clusters
        )

    if repair_enabled is not None:

        config.repair_enabled = bool(
            repair_enabled
        )

    if repair_rounds is not None:

        config.repair_rounds = int(
            repair_rounds
        )

    if max_boundary_customers is not None:

        config.max_boundary_customers = int(
            max_boundary_customers
        )

    if max_target_vehicles is not None:

        config.max_target_vehicles = int(
            max_target_vehicles
        )

    if try_swaps is not None:

        config.try_swaps = bool(
            try_swaps
        )

    if seed is not None:

        config.seed = int(
            seed
        )

    config.__post_init__()

    # ========================================================
    # RUN
    # ========================================================

    results = {}

    for algorithm in (
        available_algorithms()
    ):

        results[algorithm] = (
            run_optimizer(
                scenario=scenario,
                algorithm_name=algorithm,
                customers=customers,
                vehicles=vehicles,
                network_size=network_size,
                edges_per_node=edges_per_node,
                config=config,
            )
        )

    return results


# ============================================================
# BEST RESULT
# ============================================================

def best_result(
    results
):

    if not results:

        return None

    return min(
        results.values(),
        key=lambda result:
            result.get(
                "fitness",
                float("inf"),
            ),
    )


# ============================================================
# COMPARISON SUMMARY
# ============================================================

def comparison_summary(
    results
):

    summary = []

    for algorithm, result in (
        results.items()
    ):

        summary.append(
            {
                "algorithm": algorithm,

                "valid": result.get(
                    "valid",
                    False,
                ),

                "fitness": result.get(
                    "fitness"
                ),

                "travel_time": result.get(
                    "travel_time"
                ),

                "distance": result.get(
                    "distance"
                ),

                "congestion": result.get(
                    "congestion"
                ),

                "runtime": result.get(
                    "execution_time"
                ),
            }
        )

    return summary


# ============================================================
# SOLUTION HELPERS
# ============================================================

def get_solution_routes(
    result
):

    return result.get(
        "solution",
        {},
    )


def get_vehicle_results(
    result
):

    return result.get(
        "vehicle_results",
        {},
    )


def get_graph_nodes(
    result
):

    return result.get(
        "graph_nodes",
        [],
    )


def get_graph_edges(
    result
):

    return result.get(
        "graph_edges",
        [],
    )


# ============================================================
# RESULT SUMMARY
# ============================================================

def get_result_summary(
    result
):

    return {
        "algorithm": result.get(
            "algorithm"
        ),

        "scenario": result.get(
            "scenario"
        ),

        "valid": result.get(
            "valid"
        ),

        "fitness": result.get(
            "fitness"
        ),

        "travel_time": result.get(
            "travel_time"
        ),

        "distance": result.get(
            "distance"
        ),

        "congestion": result.get(
            "congestion"
        ),

        "execution_time": result.get(
            "execution_time"
        ),

        "customers": result.get(
            "customers"
        ),

        "vehicles": result.get(
            "vehicles"
        ),

        "network_size": result.get(
            "network_size"
        ),

        "effective_clusters": result.get(
            "effective_clusters"
        ),

        "traffic_data_source": result.get(
            "traffic_data_source"
        ),
    }


# ============================================================
# PIPELINE STEPS
# ============================================================

def get_pipeline_steps(
    algorithm_name="Decomposed Warm-QPSO",
):

    if algorithm_name == "PSO":

        return [
            "Traffic Data",
            "Large Transportation Graph",
            "VRP Construction",
            "PSO Optimization",
            "Optimized Vehicle Routes",
        ]

    if algorithm_name == "QPSO":

        return [
            "Traffic Data",
            "Large Transportation Graph",
            "VRP Construction",
            "QPSO Optimization",
            "Optimized Vehicle Routes",
        ]

    if algorithm_name == "Warm-QPSO":

        return [
            "Traffic Data",
            "Large Transportation Graph",
            "VRP Construction",
            "Classical PSO Warm Start",
            "QPSO Optimization",
            "Optimized Vehicle Routes",
        ]

    return [
        "Traffic Data",
        "Large Transportation Graph",
        "VRP Construction",
        "Problem Decomposition",
        "Classical PSO Warm Start",
        "QPSO Optimization",
        "Global Boundary Repair",
        "Optimized Vehicle Routes",
    ]


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_configuration_summary(
    config=None
):

    if config is None:

        config = OptimizationConfig()

    return config.to_dict()


# ============================================================
# TRAFFIC SUMMARY
# ============================================================

def get_traffic_summary(
    scenario="Normal",
):
    """
    Return summary information for a simulated
    traffic scenario.
    """

    records = TRAFFIC_DATA.get(
        scenario
    )

    if not records:

        return {
            "scenario": scenario,

            "records": 0,

            "average_congestion": 0.0,

            "average_speed_kmph": 0.0,

            "blocked_edges": 0,

            "data_source":
                "Simulated traffic dataset",
        }

    # --------------------------------------------------------
    # Average congestion
    # --------------------------------------------------------

    average_congestion = (
        sum(
            record["congestion"]
            for record in records
        )
        / len(records)
    )

    # --------------------------------------------------------
    # Average speed
    # --------------------------------------------------------

    valid_speeds = [
        record[
            "traffic_speed_kmph"
        ]
        for record in records
        if record[
            "traffic_speed_kmph"
        ] > 0
    ]

    if valid_speeds:

        average_speed = (
            sum(valid_speeds)
            / len(valid_speeds)
        )

    else:

        average_speed = 0.0

    # --------------------------------------------------------
    # Blocked roads
    # --------------------------------------------------------

    blocked_edges = sum(
        1
        for record in records
        if record["blocked"]
    )

    return {
        "scenario": scenario,

        "records": len(
            records
        ),

        "average_congestion":
            average_congestion,

        "average_speed_kmph":
            average_speed,

        "blocked_edges":
            blocked_edges,

        "data_source":
            "Simulated traffic dataset",
    }


# ============================================================
# TRAFFIC RECORDS
# ============================================================

def get_traffic_records(
    scenario="Normal",
):

    return TRAFFIC_DATA.get(
        scenario
    )


# ============================================================
# SCENARIO INFORMATION
# ============================================================

def get_scenario_info(
    scenario="Normal",
):

    summary = get_traffic_summary(
        scenario
    )

    return {
        "scenario": scenario,

        "traffic_source":
            "Simulated traffic dataset",

        "average_congestion":
            summary[
                "average_congestion"
            ],

        "average_speed_kmph":
            summary[
                "average_speed_kmph"
            ],

        "blocked_edges":
            summary[
                "blocked_edges"
            ],

        "records":
            summary[
                "records"
            ],
    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "Q-ROUTE OPTIMIZER SERVICE"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Algorithms
    # --------------------------------------------------------

    print(
        "Algorithms:"
    )

    print(
        available_algorithms()
    )

    print()

    # --------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------

    print(
        "Scenarios:"
    )

    print(
        available_scenarios()
    )

    print()

    # --------------------------------------------------------
    # Network sizes
    # --------------------------------------------------------

    print(
        "Network sizes:"
    )

    print(
        available_network_sizes()
    )

    print()

    # --------------------------------------------------------
    # Basic test
    # --------------------------------------------------------

    print(
        "Running 50-node test..."
    )

    result = run_optimizer(
        scenario="Normal",
        algorithm_name=(
            "Decomposed Warm-QPSO"
        ),
        customers=20,
        vehicles=5,
        network_size=50,
        particles=5,
        iterations=5,
        clusters=5,
    )

    print()

    print(
        "VALID:",
        result["valid"],
    )

    print(
        "NETWORK NODES:",
        len(
            result[
                "graph_nodes"
            ]
        ),
    )

    print(
        "NETWORK EDGES:",
        len(
            result[
                "graph_edges"
            ]
        ),
    )

    print(
        "CUSTOMERS:",
        result["customers"],
    )

    print(
        "VEHICLES:",
        result["vehicles"],
    )

    print(
        "FITNESS:",
        result["fitness"],
    )

    print(
        "TRAVEL TIME:",
        result["travel_time"],
    )

    print(
        "DISTANCE:",
        result["distance"],
    )

    print(
        "CONGESTION:",
        result["congestion"],
    )

    print(
        "RUNTIME:",
        result["execution_time"],
        "seconds",
    )

    print(
        "EFFECTIVE CLUSTERS:",
        result[
            "effective_clusters"
        ],
    )

    print()

    print("=" * 70)

    print(
        "TEST COMPLETE"
    )

    print("=" * 70)
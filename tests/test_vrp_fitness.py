import pytest

from src.domain.customer import Customer
from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem
from src.optimization.vrp_fitness import VRPFitnessFunction


# ============================================================
# TEST DATA
# ============================================================

def build_graph():
    """
    Build a small connected graph for VRP fitness testing.

    Graph:

        N1
       /  \
     N0    N3
       \  /
        N2
    """

    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")

    edges = [
        Edge(
            n0,
            n1,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n1,
            n3,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n0,
            n2,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n2,
            n3,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n1,
            n2,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n2,
            n1,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n1,
            n0,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n2,
            n0,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
        Edge(
            n3,
            n0,
            distance=5,
            travel_time=5,
            congestion=0.1,
        ),
    ]

    return Graph(
        nodes=[n0, n1, n2, n3],
        edges=edges,
    )


def build_problem():
    """
    Create a small VRP problem.

    Depot:
        N0

    Customers:
        C1 -> N1, demand 4
        C2 -> N2, demand 5
        C3 -> N3, demand 3

    Vehicles:
        V1 -> capacity 10
        V2 -> capacity 10
    """

    graph = build_graph()

    customers = [
        Customer(
            id="C1",
            demand=4,
            node_id="N1",
        ),
        Customer(
            id="C2",
            demand=5,
            node_id="N2",
        ),
        Customer(
            id="C3",
            demand=3,
            node_id="N3",
        ),
    ]

    vehicles = [
        Vehicle(
            id="V1",
            capacity=10,
            start_node="N0",
            end_node="N0",
        ),
        Vehicle(
            id="V2",
            capacity=10,
            start_node="N0",
            end_node="N0",
        ),
    ]

    return VRPProblem(
        graph=graph,
        depot="N0",
        customers=customers,
        vehicles=vehicles,
    )


# ============================================================
# EMPTY VEHICLE TESTS
# ============================================================

def test_empty_vehicle_has_zero_metrics():
    """
    An empty vehicle route is valid.

    Since the vehicle visits no customers, all route metrics
    should be zero.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        [],
    )

    assert result["valid"]
    assert result["travel_time"] == 0
    assert result["distance"] == 0
    assert result["congestion"] == 0
    assert result["fitness"] == 0
    assert result["penalty"] == 0
    assert result["customers"] == []
    assert result["demand"] == 0


def test_empty_vehicle_route():
    """
    Explicit test for an empty vehicle route.

    This directly evaluates the vehicle instead of passing an
    incomplete overall VRP solution to evaluate().
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        [],
    )

    assert result["valid"]
    assert result["travel_time"] == 0
    assert result["distance"] == 0
    assert result["congestion"] == 0


# ============================================================
# BASIC ROUTE TESTS
# ============================================================

def test_single_customer_route():
    """
    A vehicle visiting one customer and returning to the depot
    should produce a valid route.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1"],
    )

    assert result["valid"]
    assert result["travel_time"] > 0
    assert result["distance"] > 0
    assert result["congestion"] >= 0
    assert result["customers"] == ["C1"]
    assert result["demand"] == 4


def test_multiple_customer_route():
    """
    A vehicle visiting multiple customers should produce
    positive route metrics.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    assert result["valid"]
    assert result["travel_time"] > 0
    assert result["distance"] > 0
    assert result["congestion"] >= 0
    assert result["demand"] == 9


def test_customer_demand_is_calculated():
    """Route demand should equal the sum of customer demands."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    assert result["demand"] == 9


# ============================================================
# CAPACITY TESTS
# ============================================================

def test_capacity_constraint_is_satisfied():
    """A route within vehicle capacity should be valid."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    assert result["valid"]
    assert result["demand"] <= result["capacity"]


def test_capacity_constraint_is_violated():
    """
    C1 + C2 + C3 = 4 + 5 + 3 = 12.

    Vehicle capacity = 10.

    Therefore the route must be invalid.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2", "C3"],
    )

    assert not result["valid"]
    assert result["demand"] == 12
    assert result["capacity"] == 10
    assert result["penalty"] > 0
    assert result["fitness"] > 0


# ============================================================
# CUSTOMER VALIDATION TESTS
# ============================================================

def test_unknown_customer_is_invalid():
    """An unknown customer ID must be invalid."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C999"],
    )

    assert not result["valid"]
    assert result["penalty"] > 0


def test_duplicate_customer_in_route_is_invalid():
    """
    The same customer cannot appear twice in a complete
    VRP solution.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1", "C1"],
        "V2": ["C2", "C3"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]


def test_unknown_vehicle_is_invalid():
    """An unknown vehicle ID should be rejected."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V999",
        ["C1"],
    )

    assert not result["valid"]
    assert result["penalty"] > 0


def test_vehicle_object_can_be_used():
    """
    evaluate_vehicle_route() should accept a Vehicle object
    as well as a vehicle ID.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    vehicle = problem.vehicles[0]

    result = fitness.evaluate_vehicle_route(
        problem,
        vehicle,
        ["C1"],
    )

    assert result["valid"]
    assert result["customers"] == ["C1"]


# ============================================================
# PHYSICAL ROUTE TESTS
# ============================================================

def test_route_returns_to_depot():
    """
    The physical route should start at the vehicle's start node
    and finish at its end node.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    assert result["valid"]
    assert result["route"][0].id == "N0"
    assert result["route"][-1].id == "N0"


# ============================================================
# COMPLETE VRP SOLUTION TESTS
# ============================================================

def test_valid_complete_solution():
    """
    Every customer is assigned exactly once and vehicle
    capacities are respected.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1"],
        "V2": ["C2", "C3"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert result["valid"]
    assert result["travel_time"] > 0
    assert result["distance"] > 0
    assert result["fitness"] > 0
    assert len(result["vehicle_results"]) == 2


def test_solution_with_missing_customer_is_invalid():
    """
    A complete VRP solution must assign every customer.

    C3 is intentionally missing.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1"],
        "V2": ["C2"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]
    assert result["fitness"] >= fitness.invalid_penalty


def test_solution_with_duplicate_customer_is_invalid():
    """
    A customer cannot be assigned to multiple vehicles.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1", "C2"],
        "V2": ["C2", "C3"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]
    assert result["fitness"] >= fitness.invalid_penalty


def test_solution_with_unknown_customer_is_invalid():
    """An unknown customer ID invalidates the complete solution."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1", "C999"],
        "V2": ["C2", "C3"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]


def test_solution_with_unknown_vehicle_is_invalid():
    """An unknown vehicle ID invalidates the complete solution."""

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1"],
        "V2": ["C2", "C3"],
        "V999": [],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]


def test_solution_missing_vehicle_is_invalid():
    """
    A solution must contain exactly the vehicles defined by
    the VRP problem.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1", "C2"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    assert not result["valid"]


# ============================================================
# FITNESS WEIGHT TESTS
# ============================================================

def test_fitness_weights_are_normalized():
    """
    Fitness weights should be normalized so their sum is 1.
    """

    fitness = VRPFitnessFunction(
        time_weight=2.0,
        distance_weight=1.0,
        congestion_weight=1.0,
    )

    total = (
        fitness.time_weight
        + fitness.distance_weight
        + fitness.congestion_weight
    )

    assert total == pytest.approx(1.0)

    assert fitness.time_weight == pytest.approx(0.5)
    assert fitness.distance_weight == pytest.approx(0.25)
    assert fitness.congestion_weight == pytest.approx(0.25)


def test_zero_weight_sum_is_rejected():
    """All fitness weights cannot be zero."""

    with pytest.raises(ValueError):
        VRPFitnessFunction(
            time_weight=0,
            distance_weight=0,
            congestion_weight=0,
        )


# ============================================================
# FITNESS BEHAVIOR TESTS
# ============================================================

def test_lower_travel_time_route_has_lower_fitness():
    """
    When travel time is the only weighted objective, a route
    with lower travel time should have lower fitness.
    """

    problem = build_problem()

    fitness = VRPFitnessFunction(
        time_weight=1.0,
        distance_weight=0.0,
        congestion_weight=0.0,
    )

    result_1 = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1"],
    )

    result_2 = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    assert result_1["travel_time"] < result_2["travel_time"]
    assert result_1["fitness"] < result_2["fitness"]


def test_capacity_penalty_increases_fitness():
    """
    A capacity violation should introduce a penalty.
    """

    problem = build_problem()

    fitness = VRPFitnessFunction(
        capacity_penalty=1000,
    )

    valid_result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    invalid_result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2", "C3"],
    )

    assert valid_result["valid"]
    assert not invalid_result["valid"]
    assert invalid_result["penalty"] > 0
    assert invalid_result["fitness"] > valid_result["fitness"]


# ============================================================
# EDGE INPUT VALIDATION TESTS
# ============================================================

def test_negative_travel_time_is_rejected():
    """
    Edge creation should reject negative travel time.
    """

    n0 = Node("N0")
    n1 = Node("N1")

    with pytest.raises(ValueError):
        Edge(
            n0,
            n1,
            distance=5,
            travel_time=-1,
            congestion=0.1,
        )


def test_negative_distance_is_rejected():
    """
    Edge creation should reject negative distance.
    """

    n0 = Node("N0")
    n1 = Node("N1")

    with pytest.raises(ValueError):
        Edge(
            n0,
            n1,
            distance=-1,
            travel_time=5,
            congestion=0.1,
        )


def test_negative_congestion_is_rejected():
    """
    Edge creation should reject negative congestion.
    """

    n0 = Node("N0")
    n1 = Node("N1")

    with pytest.raises(ValueError):
        Edge(
            n0,
            n1,
            distance=5,
            travel_time=5,
            congestion=-0.1,
        )


# ============================================================
# CONGESTION TESTS
# ============================================================

def test_congestion_is_capped():
    """
    Congestion should remain within the expected range [0, 1].
    """

    problem = build_problem()

    for edge in problem.graph.get_edges():
        edge.congestion = 1.0

    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1"],
    )

    assert result["valid"]
    assert result["congestion"] <= 1.0


def test_congestion_affects_fitness():
    """
    Higher congestion should increase fitness when congestion
    is the only weighted objective.
    """

    problem_low = build_problem()
    problem_high = build_problem()

    for edge in problem_low.graph.get_edges():
        edge.congestion = 0.1

    for edge in problem_high.graph.get_edges():
        edge.congestion = 0.9

    fitness = VRPFitnessFunction(
        time_weight=0.0,
        distance_weight=0.0,
        congestion_weight=1.0,
    )

    low_result = fitness.evaluate_vehicle_route(
        problem_low,
        "V1",
        ["C1"],
    )

    high_result = fitness.evaluate_vehicle_route(
        problem_high,
        "V1",
        ["C1"],
    )

    assert low_result["valid"]
    assert high_result["valid"]

    assert high_result["congestion"] > low_result["congestion"]
    assert high_result["fitness"] > low_result["fitness"]


# ============================================================
# CUSTOM PENALTY TEST
# ============================================================

def test_custom_penalty_is_added():
    """
    A capacity violation should add the configured penalty.
    """

    problem = build_problem()

    fitness = VRPFitnessFunction(
        capacity_penalty=100,
    )

    valid_result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2"],
    )

    invalid_result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1", "C2", "C3"],
    )

    assert valid_result["valid"]
    assert not invalid_result["valid"]

    assert invalid_result["penalty"] > 0
    assert invalid_result["fitness"] > valid_result["fitness"]


# ============================================================
# UNREACHABLE ROUTE TEST
# ============================================================

def test_unreachable_customer_route_is_invalid():
    """
    If a customer cannot be reached from the vehicle's current
    position, the route should be invalid.
    """

    problem = build_problem()

    # Block every edge whose target is N3.
    for edge in problem.graph.get_edges():
        if edge.target.id == "N3":
            edge.blocked = True

    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C3"],
    )

    assert not result["valid"]
    assert result["penalty"] > 0


# ============================================================
# RESULT STRUCTURE TESTS
# ============================================================

def test_vehicle_result_contains_expected_fields():
    """
    Vehicle evaluation should return all expected fields.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    result = fitness.evaluate_vehicle_route(
        problem,
        "V1",
        ["C1"],
    )

    expected_fields = {
        "valid",
        "travel_time",
        "distance",
        "congestion",
        "fitness",
        "penalty",
        "route",
        "customers",
        "demand",
        "capacity",
    }

    assert expected_fields.issubset(result.keys())


def test_solution_result_contains_expected_fields():
    """
    Complete solution evaluation should return all expected fields.
    """

    problem = build_problem()
    fitness = VRPFitnessFunction()

    solution = {
        "V1": ["C1"],
        "V2": ["C2", "C3"],
    }

    result = fitness.evaluate(
        problem,
        solution,
    )

    expected_fields = {
        "valid",
        "travel_time",
        "distance",
        "congestion",
        "fitness",
        "penalty",
        "solution",
        "vehicle_results",
    }

    assert expected_fields.issubset(result.keys())
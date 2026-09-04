from src.domain.customer import Customer
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem
from src.algorithms.vrp_route_utils import VRPRouteUtils


def build_problem():
    customers = [
        Customer("C1", 4, "N1"),
        Customer("C2", 7, "N2"),
        Customer("C3", 3, "N3"),
        Customer("C4", 5, "N4"),
        Customer("C5", 6, "N5"),
    ]

    vehicles = [
        Vehicle(
            "V1",
            capacity=15,
            start_node="N0",
            end_node="N0",
        ),
        Vehicle(
            "V2",
            capacity=15,
            start_node="N0",
            end_node="N0",
        ),
    ]

    return VRPProblem(
        graph=None,
        depot="N0",
        customers=customers,
        vehicles=vehicles,
    )


def test_empty_solution():
    problem = build_problem()

    solution = VRPRouteUtils.empty_solution(problem)

    assert solution == {
        "V1": [],
        "V2": [],
    }


def test_customer_assignment():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C3"],
        "V2": ["C2", "C4", "C5"],
    }

    assert VRPRouteUtils.customer_assignment_valid(
        problem,
        solution,
    )


def test_duplicate_customer_is_invalid():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C2"],
        "V2": ["C2", "C3", "C4", "C5"],
    }

    assert not VRPRouteUtils.customer_assignment_valid(
        problem,
        solution,
    )


def test_missing_customer_is_invalid():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C3"],
        "V2": ["C2", "C4"],
    }

    assert not VRPRouteUtils.customer_assignment_valid(
        problem,
        solution,
    )


def test_vehicle_demand():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C3"],
        "V2": ["C2", "C4", "C5"],
    }

    assert (
        VRPRouteUtils.vehicle_route_demand(
            problem,
            solution,
            "V1",
        )
        == 7
    )


def test_capacity_constraint():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C2", "C3"],
        "V2": ["C4", "C5"],
    }

    assert VRPRouteUtils.vehicle_capacity_feasible(
        problem,
        solution,
        "V1",
    )

    assert VRPRouteUtils.vehicle_capacity_feasible(
        problem,
        solution,
        "V2",
    )


def test_capacity_violation():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C2", "C4"],
        "V2": ["C3", "C5"],
    }

    assert not VRPRouteUtils.vehicle_capacity_feasible(
        problem,
        solution,
        "V1",
    )


def test_complete_solution():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C2"],
        "V2": ["C3", "C4", "C5"],
    }

    assert VRPRouteUtils.is_valid_solution(
        problem,
        solution,
    )


def test_invalid_solution():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C2", "C4"],
        "V2": ["C2", "C3", "C5"],
    }

    assert not VRPRouteUtils.is_valid_solution(
        problem,
        solution,
    )


def test_greedy_assignment():
    problem = build_problem()

    solution = (
        VRPRouteUtils.assign_customers_greedily(
            problem
        )
    )

    assert solution is not None

    assert VRPRouteUtils.is_valid_solution(
        problem,
        solution,
    )


def test_flatten_solution():
    problem = build_problem()

    solution = {
        "V1": ["C1", "C3"],
        "V2": ["C2", "C4", "C5"],
    }

    flattened = (
        VRPRouteUtils.flatten_solution(
            problem,
            solution,
        )
    )

    assert flattened == [
        ("V1", "C1"),
        ("V1", "C3"),
        ("V2", "C2"),
        ("V2", "C4"),
        ("V2", "C5"),
    ]
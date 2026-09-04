from src.algorithms.vrp_pso import VRPPSO

from src.domain.customer import Customer
from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem


def build_graph():
    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")

    edges = [
        Edge(n0, n1, 5, 5, 0.1),
        Edge(n1, n0, 5, 5, 0.1),

        Edge(n0, n2, 5, 5, 0.1),
        Edge(n2, n0, 5, 5, 0.1),

        Edge(n0, n3, 8, 8, 0.1),
        Edge(n3, n0, 8, 8, 0.1),

        Edge(n1, n2, 5, 5, 0.1),
        Edge(n2, n1, 5, 5, 0.1),

        Edge(n1, n3, 5, 5, 0.1),
        Edge(n3, n1, 5, 5, 0.1),

        Edge(n2, n3, 5, 5, 0.1),
        Edge(n3, n2, 5, 5, 0.1),
    ]

    return Graph(
        nodes=[n0, n1, n2, n3],
        edges=edges,
    )


def build_problem():
    graph = build_graph()

    customers = [
        Customer("C1", 4, "N1"),
        Customer("C2", 3, "N2"),
        Customer("C3", 2, "N3"),
    ]

    vehicles = [
        Vehicle("V1", 6, "N0", "N0"),
        Vehicle("V2", 6, "N0", "N0"),
    ]

    return VRPProblem(
        graph=graph,
        depot="N0",
        customers=customers,
        vehicles=vehicles,
    )


def test_vrp_pso_initialization():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=10,
        iterations=5,
        seed=42,
    )

    assert optimizer.particles == 10
    assert optimizer.iterations == 5
    assert optimizer.dimension == 3


def test_random_position():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = optimizer._random_position()

    assert len(position) == 3

    for value in position:
        assert 0.0 <= value <= 1.0


def test_customer_order_decoding():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = [
        0.9,
        0.1,
        0.5,
    ]

    order = optimizer._decode_customer_order(
        position
    )

    assert len(order) == 3
    assert set(order) == {
        "C1",
        "C2",
        "C3",
    }


def test_position_decoding_returns_solution():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = [
        0.2,
        0.8,
        0.5,
    ]

    solution = optimizer.decode_position(
        position
    )

    assert solution is not None

    assert set(solution.keys()) == {
        "V1",
        "V2",
    }


def test_all_customers_are_assigned():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = [
        0.2,
        0.8,
        0.5,
    ]

    solution = optimizer.decode_position(
        position
    )

    assigned = []

    for route in solution.values():
        assigned.extend(route)

    assert sorted(assigned) == [
        "C1",
        "C2",
        "C3",
    ]


def test_vehicle_capacity_is_respected():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = [
        0.2,
        0.8,
        0.5,
    ]

    solution = optimizer.decode_position(
        position
    )

    for vehicle in problem.vehicles:

        demand = sum(
            problem.customer_map[c].demand
            for c in solution[vehicle.id]
        )

        assert demand <= vehicle.capacity


def test_evaluate_position():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=5,
        iterations=2,
        seed=42,
    )

    position = [
        0.2,
        0.8,
        0.5,
    ]

    result = optimizer.evaluate_position(
        position
    )

    assert result["valid"]
    assert result["fitness"] >= 0
    assert result["travel_time"] > 0
    assert result["distance"] > 0


def test_pso_returns_result():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=10,
        iterations=10,
        seed=42,
    )

    result = optimizer.optimize()

    assert result["solution"] is not None
    assert result["fitness"] >= 0
    assert result["travel_time"] > 0
    assert result["distance"] > 0


def test_pso_solution_is_complete():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=10,
        iterations=10,
        seed=42,
    )

    result = optimizer.optimize()

    solution = result["solution"]

    assigned = []

    for route in solution.values():
        assigned.extend(route)

    assert sorted(assigned) == [
        "C1",
        "C2",
        "C3",
    ]


def test_pso_history():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=10,
        iterations=10,
        seed=42,
    )

    result = optimizer.optimize()

    assert len(result["history"]) == 11

    for value in result["history"]:
        assert value >= 0


def test_pso_history_is_non_increasing():
    problem = build_problem()

    optimizer = VRPPSO(
        problem,
        particles=10,
        iterations=10,
        seed=42,
    )

    result = optimizer.optimize()

    history = result["history"]

    for previous, current in zip(
        history,
        history[1:],
    ):
        assert current <= previous


def test_pso_reproducibility():
    problem1 = build_problem()
    problem2 = build_problem()

    optimizer1 = VRPPSO(
        problem1,
        particles=10,
        iterations=10,
        seed=42,
    )

    optimizer2 = VRPPSO(
        problem2,
        particles=10,
        iterations=10,
        seed=42,
    )

    result1 = optimizer1.optimize()
    result2 = optimizer2.optimize()

    assert result1["fitness"] == result2["fitness"]
    assert result1["solution"] == result2["solution"]
    assert result1["history"] == result2["history"]
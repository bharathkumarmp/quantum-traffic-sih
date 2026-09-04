from src.algorithms.vrp_qpso import VRPQPSO
from src.domain.customer import Customer
from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem


def create_test_problem():
    """
    Create a small deterministic capacitated VRP
    problem for testing VRPQPSO.

    Network:

        N0 ---- N1 ---- N2
         |       |       |
         +-------+-------+

    Depot:
        N0

    Customers:
        C1 -> N1
        C2 -> N2
        C3 -> N1

    Vehicles:
        V1 capacity = 10
        V2 capacity = 10
    """

    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")

    edges = [
        Edge(
            n0,
            n1,
            distance=5.0,
            travel_time=5.0,
            congestion=0.1,
        ),
        Edge(
            n1,
            n0,
            distance=5.0,
            travel_time=5.0,
            congestion=0.1,
        ),
        Edge(
            n1,
            n2,
            distance=4.0,
            travel_time=4.0,
            congestion=0.1,
        ),
        Edge(
            n2,
            n1,
            distance=4.0,
            travel_time=4.0,
            congestion=0.1,
        ),
        Edge(
            n0,
            n2,
            distance=8.0,
            travel_time=8.0,
            congestion=0.1,
        ),
        Edge(
            n2,
            n0,
            distance=8.0,
            travel_time=8.0,
            congestion=0.1,
        ),
    ]

    graph = Graph(
        nodes=[n0, n1, n2],
        edges=edges,
    )

    customers = [
        Customer(
            id="C1",
            demand=3.0,
            node_id="N1",
        ),
        Customer(
            id="C2",
            demand=4.0,
            node_id="N2",
        ),
        Customer(
            id="C3",
            demand=2.0,
            node_id="N1",
        ),
    ]

    vehicles = [
        Vehicle(
            id="V1",
            capacity=10.0,
            start_node="N0",
            end_node="N0",
        ),
        Vehicle(
            id="V2",
            capacity=10.0,
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


def test_vrp_qpso_runs():
    problem = create_test_problem()

    optimizer = VRPQPSO(
        problem,
        particles=10,
        iterations=10,
        beta=0.7,
        seed=42,
    )

    result = optimizer.optimize()

    assert result["valid"] is True
    assert result["solution"] is not None

    assert result["fitness"] >= 0
    assert result["travel_time"] >= 0
    assert result["distance"] >= 0
    assert result["congestion"] >= 0

    assert len(result["history"]) == 11

    assert result["iterations"] == 10
    assert result["particles"] == 10
    assert result["beta"] == 0.7


def test_qpso_customer_assignment():
    problem = create_test_problem()

    optimizer = VRPQPSO(
        problem,
        particles=10,
        iterations=10,
        beta=0.7,
        seed=42,
    )

    result = optimizer.optimize()

    solution = result["solution"]

    assigned_customers = []

    for customer_ids in solution.values():
        assigned_customers.extend(
            customer_ids
        )

    expected_customers = sorted(
        customer.id
        for customer in problem.customers
    )

    assert sorted(
        assigned_customers
    ) == expected_customers


def test_qpso_position_update_stays_in_bounds():
    problem = create_test_problem()

    optimizer = VRPQPSO(
        problem,
        particles=5,
        iterations=5,
        beta=0.7,
        seed=42,
    )

    position = [
        0.2
        for _ in range(
            optimizer.dimension
        )
    ]

    best_position = [
        0.5
        for _ in range(
            optimizer.dimension
        )
    ]

    mean_best = [
        0.4
        for _ in range(
            optimizer.dimension
        )
    ]

    new_position = optimizer._update_position(
        position,
        best_position,
        mean_best,
    )

    assert len(
        new_position
    ) == optimizer.dimension

    for value in new_position:
        assert 0.0 <= value <= 1.0
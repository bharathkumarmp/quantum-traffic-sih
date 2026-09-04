from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.route import Route
from src.optimization.fitness import FitnessFunction


def test_graph():
    n0 = Node("N0")
    n1 = Node("N1")
    n5 = Node("N5")

    e01 = Edge(
        n0,
        n1,
        distance=10,
        travel_time=12,
        congestion=0.2,
    )

    e15 = Edge(
        n1,
        n5,
        distance=10,
        travel_time=12,
        congestion=0.2,
    )

    graph = Graph()

    graph.add_edge(e01)
    graph.add_edge(e15)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    neighbors = graph.get_neighbors(n0)

    assert len(neighbors) == 1
    assert neighbors[0].target == n1


def test_route():
    n0 = Node("N0")
    n1 = Node("N1")
    n5 = Node("N5")

    route = Route([n0, n1, n5])

    assert route.start == n0
    assert route.destination == n5
    assert len(route) == 3


def test_fitness():
    fitness = FitnessFunction()

    value = fitness.calculate(
        travel_time=24,
        distance=20,
        congestion=0.2,
    )

    assert value > 0

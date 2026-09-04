from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.traffic.scenario import (
    TrafficScenario,
)

from src.traffic.traffic_model import (
    TrafficModel,
)


def build_graph():

    n0 = Node("N0")
    n1 = Node("N1")
    n2 = Node("N2")

    graph = Graph()

    graph.add_edge(
        Edge(
            n0,
            n1,
            distance=10,
            travel_time=10,
            congestion=0.2,
        )
    )

    graph.add_edge(
        Edge(
            n1,
            n2,
            distance=10,
            travel_time=10,
            congestion=0.3,
        )
    )

    return graph


def test_scenario_creation():

    scenario = TrafficScenario(
        "test"
    )

    assert scenario.name == "test"

    assert len(
        scenario.updates
    ) == 0


def test_scenario_update():

    scenario = TrafficScenario(
        "incident"
    )

    scenario.add_update(
        "N0",
        "N1",
        travel_time=30,
        congestion=0.9,
    )

    update = scenario.get_update(
        "N0",
        "N1",
    )

    assert update is not None

    assert (
        update["travel_time"]
        == 30
    )

    assert (
        update["congestion"]
        == 0.9
    )


def test_scenario_apply():

    graph = build_graph()

    scenario = TrafficScenario(
        "incident"
    )

    scenario.add_update(
        "N0",
        "N1",
        travel_time=30,
        congestion=0.9,
    )

    scenario.apply(
        graph
    )

    edge = graph.get_edges()[0]

    assert (
        edge.travel_time
        == 30
    )

    assert (
        edge.congestion
        == 0.9
    )


def test_heavy_traffic():

    graph = build_graph()

    model = TrafficModel(
        graph
    )

    scenario = (
        model.heavy_traffic_scenario()
    )

    scenario.apply(
        graph
    )

    for edge in graph.get_edges():

        assert (
            edge.travel_time
            > 0
        )

        assert (
            0
            <= edge.congestion
            <= 1
        )


def test_incident():

    graph = build_graph()

    model = TrafficModel(
        graph
    )

    scenario = (
        model.incident_scenario(
            "N0",
            "N1",
        )
    )

    scenario.apply(
        graph
    )

    edge = graph.get_edges()[0]

    assert (
        edge.travel_time
        == 30
    )

    assert (
        edge.congestion
        == 0.95
    )


def test_road_blockage():

    graph = build_graph()

    model = TrafficModel(
        graph
    )

    scenario = (
        model.road_blockage_scenario(
            "N0",
            "N1",
        )
    )

    scenario.apply(
        graph
    )

    edge = graph.get_edges()[0]

    assert edge.blocked is True
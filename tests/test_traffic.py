from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph

from src.traffic.scenario import (
    TrafficScenario,
    create_scenario,
)

from src.traffic.traffic_model import TrafficModel


def build_test_graph():
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


def test_normal_scenario():
    scenario = create_scenario(
        TrafficScenario.NORMAL
    )

    assert scenario.congestion_multiplier == 1.0


def test_peak_increases_congestion():
    graph = build_test_graph()

    model = TrafficModel(graph)

    scenario = create_scenario(
        TrafficScenario.PEAK
    )

    updated = model.apply_scenario(scenario)

    original_edge = graph.get_edges()[0]
    updated_edge = updated.get_edges()[0]

    assert updated_edge.congestion > original_edge.congestion
    assert updated_edge.travel_time > original_edge.travel_time


def test_original_graph_is_not_modified():
    graph = build_test_graph()

    original_congestion = graph.get_edges()[0].congestion

    model = TrafficModel(graph)

    scenario = create_scenario(
        TrafficScenario.CONGESTED
    )

    model.apply_scenario(scenario)

    assert (
        graph.get_edges()[0].congestion
        == original_congestion
    )


def test_incident_blocks_edge():
    graph = build_test_graph()

    model = TrafficModel(graph)

    success = model.apply_edge_incident(
        graph,
        "N0",
        "N1",
    )

    assert success is True
    assert graph.get_edges()[0].blocked is True

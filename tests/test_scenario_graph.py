from src.data.scenario_graph_generator import (
    ScenarioGraphGenerator,
)


def test_scenario_graph_has_two_routes():

    generator = ScenarioGraphGenerator()

    graph = generator.generate()

    assert len(graph.get_edges()) == 6


def test_upper_route_incident():

    generator = ScenarioGraphGenerator()

    graph = generator.upper_route_incident()

    affected_edges = [
        edge
        for edge in graph.get_edges()
        if edge.source.id == "N1"
        and edge.target.id == "N2"
    ]

    assert len(affected_edges) == 1

    edge = affected_edges[0]

    assert edge.travel_time == 30.0
    assert edge.congestion == 0.95


def test_upper_route_blockage():

    generator = ScenarioGraphGenerator()

    graph = generator.upper_route_blockage()

    affected_edges = [
        edge
        for edge in graph.get_edges()
        if edge.source.id == "N1"
        and edge.target.id == "N2"
    ]

    assert len(affected_edges) == 1

    assert affected_edges[0].blocked is True


def test_lower_route_incident():

    generator = ScenarioGraphGenerator()

    graph = generator.lower_route_incident()

    affected_edges = [
        edge
        for edge in graph.get_edges()
        if edge.source.id == "N3"
        and edge.target.id == "N4"
    ]

    assert len(affected_edges) == 1

    edge = affected_edges[0]

    assert edge.travel_time == 30.0
    assert edge.congestion == 0.95
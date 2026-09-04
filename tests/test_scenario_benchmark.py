from src.optimization.scenario_benchmark import ScenarioBenchmark


def test_scenario_benchmark():
    benchmark = ScenarioBenchmark(
        particles=20,
        iterations=50,
        seed=42,
    )

    results = benchmark.run()

    assert len(results) == 12

    scenarios = {
        result["scenario"]
        for result in results
    }

    algorithms = {
        result["algorithm"]
        for result in results
    }

    assert scenarios == {
        "normal",
        "upper_route_incident",
        "upper_route_blockage",
        "lower_route_incident",
    }

    assert algorithms == {
        "Dijkstra",
        "PSO",
        "QPSO",
    }

    for result in results:
        assert result["route"][0] == "N0"
        assert result["route"][-1] == "N5"
        assert result["fitness"] >= 0
        assert result["travel_time"] >= 0
        assert result["distance"] >= 0
        assert result["congestion"] >= 0
        assert result["execution_time"] >= 0
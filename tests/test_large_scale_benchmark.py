from src.optimization.large_scale_benchmark import LargeScaleBenchmark


def test_large_scale_benchmark():
    benchmark = LargeScaleBenchmark(
        network_sizes=(20, 30),
        particles=5,
        iterations=5,
        seed=42,
    )

    results = benchmark.run()

    assert len(results) == 6

    algorithms = {
        result["algorithm"]
        for result in results
    }

    assert algorithms == {
        "Dijkstra",
        "PSO",
        "QPSO",
    }

    for result in results:
        assert result["fitness"] >= 0
        assert result["travel_time"] >= 0
        assert result["distance"] >= 0
        assert result["congestion"] >= 0
        assert result["execution_time"] >= 0
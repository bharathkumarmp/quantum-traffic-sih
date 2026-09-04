from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.data.graph_generator import LargeGraphGenerator
from src.optimization.benchmark import Benchmark


def test_benchmark_runs_all_algorithms():

    graph = LargeGraphGenerator(
        num_nodes=20,
        edges_per_node=3,
        seed=42,
    ).generate()

    dijkstra = Dijkstra(graph)

    pso = PSO(
        graph,
        particles=5,
        iterations=5,
        seed=42,
    )

    qpso = QPSO(
        graph,
        particles=5,
        iterations=5,
        seed=42,
    )

    benchmark = Benchmark(
        graph,
        {
            "Dijkstra": dijkstra,
            "PSO": pso,
            "QPSO": qpso,
        },
    )

    results = benchmark.run()

    assert len(results) == 3

    names = {result.algorithm for result in results}

    assert names == {
        "Dijkstra",
        "PSO",
        "QPSO",
    }


def test_benchmark_result_contains_metrics():

    graph = LargeGraphGenerator(
        num_nodes=10,
        edges_per_node=2,
        seed=42,
    ).generate()

    pso = PSO(
        graph,
        particles=5,
        iterations=3,
        seed=42,
    )

    benchmark = Benchmark(
        graph,
        {
            "PSO": pso,
        },
    )

    results = benchmark.run()

    result = results[0]

    assert result.algorithm == "PSO"
    assert result.fitness >= 0
    assert result.travel_time >= 0
    assert result.distance >= 0
    assert result.congestion >= 0
    assert result.execution_time >= 0


def test_benchmark_summary():

    graph = LargeGraphGenerator(
        num_nodes=10,
        edges_per_node=2,
        seed=42,
    ).generate()

    pso = PSO(
        graph,
        particles=5,
        iterations=3,
        seed=42,
    )

    benchmark = Benchmark(
        graph,
        {
            "PSO": pso,
        },
    )

    results = benchmark.run()

    summary = benchmark.summarize(results)

    assert len(summary) == 1
    assert summary[0]["algorithm"] == "PSO"
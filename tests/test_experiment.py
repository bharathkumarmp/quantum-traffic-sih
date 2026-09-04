from pathlib import Path

from src.optimization.experiment import TrafficExperiment


def test_experiment_runs_all_scenarios():

    experiment = TrafficExperiment(
        num_nodes=15,
        edges_per_node=3,
        particles=5,
        iterations=5,
        seed=42,
    )

    results = experiment.run_all()

    assert len(results) == 15

    scenarios = {
        result["scenario"]
        for result in results
    }

    assert scenarios == {
        "normal",
        "peak",
        "congested",
        "incident",
        "road_blockage",
    }


def test_experiment_contains_all_algorithms():

    experiment = TrafficExperiment(
        num_nodes=15,
        edges_per_node=3,
        particles=5,
        iterations=5,
        seed=42,
    )

    results = experiment.run_scenario(
        "normal"
    )

    algorithms = {
        result["algorithm"]
        for result in results
    }

    assert algorithms == {
        "Dijkstra",
        "PSO",
        "QPSO",
    }


def test_results_can_be_saved(tmp_path):

    experiment = TrafficExperiment(
        num_nodes=10,
        edges_per_node=2,
        particles=3,
        iterations=3,
        seed=42,
    )

    results = experiment.run_scenario(
        "normal"
    )

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"

    experiment.save_json(
        results,
        json_path,
    )

    experiment.save_csv(
        results,
        csv_path,
    )

    assert json_path.exists()
    assert csv_path.exists()
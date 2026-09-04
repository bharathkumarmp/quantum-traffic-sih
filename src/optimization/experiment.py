import csv
import json
import time
from pathlib import Path

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.data.graph_generator import LargeGraphGenerator
from src.optimization.fitness import FitnessFunction
from src.traffic.scenario import TrafficScenario


class TrafficExperiment:
    """
    Reproducible comparison of Dijkstra, PSO and QPSO
    under controlled traffic scenarios.
    """

    SCENARIOS = (
        "normal",
        "peak",
        "congested",
        "incident",
        "road_blockage",
    )

    ALGORITHMS = (
        "Dijkstra",
        "PSO",
        "QPSO",
    )

    def __init__(
        self,
        num_nodes=30,
        edges_per_node=3,
        particles=10,
        iterations=20,
        seed=42,
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")

        if edges_per_node < 1:
            raise ValueError(
                "edges_per_node must be at least 1."
            )

        if particles < 2:
            raise ValueError(
                "particles must be at least 2."
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be at least 1."
            )

        self.num_nodes = int(num_nodes)
        self.edges_per_node = int(edges_per_node)
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.seed = int(seed)

        # SAME objective for every algorithm.
        self.fitness_function = FitnessFunction(
            time_weight=0.5,
            distance_weight=0.3,
            congestion_weight=0.2,
        )

    def _create_graph(self):
        return LargeGraphGenerator(
            num_nodes=self.num_nodes,
            edges_per_node=self.edges_per_node,
            seed=self.seed,
        ).generate()

    @staticmethod
    def _node_ids(graph):
        if hasattr(graph, "_node_map"):
            return list(graph._node_map.keys())

        if hasattr(graph, "nodes"):
            if isinstance(graph.nodes, dict):
                return list(graph.nodes.keys())

            return [
                node.id if hasattr(node, "id") else str(node)
                for node in graph.nodes
            ]

        raise ValueError(
            "Unable to retrieve graph nodes."
        )

    def _create_scenario(
        self,
        scenario_name,
        nodes,
    ):
        """
        Create a controlled traffic scenario.

        N1 -> N2 is guaranteed to exist because the
        graph generator creates the sequential backbone.
        """

        affected_source = nodes[1]
        affected_target = nodes[2]

        if scenario_name == "normal":
            return TrafficScenario("normal")

        if scenario_name == "peak":
            return TrafficScenario("peak")

        if scenario_name == "congested":
            return TrafficScenario("congested")

        if scenario_name == "incident":
            return TrafficScenario(
                "incident",
                source=affected_source,
                target=affected_target,
                travel_time_multiplier=3.0,
                congestion=0.95,
                blocked=False,
            )

        if scenario_name == "road_blockage":
            return TrafficScenario(
                "road_blockage",
                source=affected_source,
                target=affected_target,
                blocked=True,
            )

        raise ValueError(
            f"Unknown scenario: {scenario_name}"
        )

    def _run_algorithm(
        self,
        name,
        graph,
        start,
        destination,
    ):
        if name == "Dijkstra":
            algorithm = Dijkstra(
                graph,
                fitness_function=self.fitness_function,
            )

        elif name == "PSO":
            algorithm = PSO(
                graph,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
            )

        elif name == "QPSO":
            algorithm = QPSO(
                graph,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {name}"
            )

        start_time = time.perf_counter()

        result = algorithm.optimize(
            start,
            destination,
        )

        runtime = time.perf_counter() - start_time

        return {
            "algorithm": name,
            "fitness": float(result["fitness"]),
            "travel_time": float(
                result["travel_time"]
            ),
            "distance": float(
                result["distance"]
            ),
            "congestion": float(
                result["congestion"]
            ),
            "runtime": float(runtime),
            "iterations": int(
                result.get("iterations", 0)
            ),
        }

    def run_scenario(self, scenario_name):
        """
        Run every algorithm on the same graph
        and same traffic scenario.
        """

        if scenario_name not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_name}"
            )

        # Create one base graph.
        base_graph = self._create_graph()

        nodes = self._node_ids(base_graph)

        start = nodes[0]
        destination = nodes[-1]

        # Create a fresh copy through scenario application.
        scenario = self._create_scenario(
            scenario_name,
            nodes,
        )

        graph = scenario.apply(base_graph)

        results = []

        for algorithm_name in self.ALGORITHMS:

            try:
                result = self._run_algorithm(
                    algorithm_name,
                    graph,
                    start,
                    destination,
                )

                result["scenario"] = scenario_name
                result["nodes"] = self.num_nodes
                result["edges_per_node"] = (
                    self.edges_per_node
                )
                result["seed"] = self.seed

                results.append(result)

            except (ValueError, KeyError):

                results.append({
                    "algorithm": algorithm_name,
                    "scenario": scenario_name,
                    "nodes": self.num_nodes,
                    "edges_per_node": (
                        self.edges_per_node
                    ),
                    "fitness": float("inf"),
                    "travel_time": float("inf"),
                    "distance": float("inf"),
                    "congestion": float("inf"),
                    "runtime": 0.0,
                    "iterations": 0,
                    "seed": self.seed,
                })

        return results

    def run_all(self):
        """Run every configured traffic scenario."""

        results = []

        for scenario in self.SCENARIOS:
            results.extend(
                self.run_scenario(scenario)
            )

        return results

    @staticmethod
    def save_json(results, path):
        """Save benchmark results as JSON."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
            )

    @staticmethod
    def save_csv(results, path):
        """Save benchmark results as CSV."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not results:
            return

        fieldnames = list(
            results[0].keys()
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)


def print_results(results):
    """Print benchmark results in a readable table."""

    print()
    print("Traffic Benchmark Results")
    print("=" * 105)

    header = (
        f"{'Scenario':15}"
        f"{'Algorithm':12}"
        f"{'Fitness':12}"
        f"{'Time':12}"
        f"{'Distance':12}"
        f"{'Congestion':12}"
        f"{'Runtime':12}"
    )

    print(header)
    print("-" * 105)

    for result in results:

        print(
            f"{result['scenario']:15}"
            f"{result['algorithm']:12}"
            f"{result['fitness']:12.4f}"
            f"{result['travel_time']:12.4f}"
            f"{result['distance']:12.4f}"
            f"{result['congestion']:12.4f}"
            f"{result['runtime']:12.6f}"
        )


if __name__ == "__main__":

    experiment = TrafficExperiment(
        num_nodes=30,
        edges_per_node=3,
        particles=10,
        iterations=20,
        seed=42,
    )

    results = experiment.run_all()

    output_dir = Path("results")

    experiment.save_json(
        results,
        output_dir / "traffic_benchmark.json",
    )

    experiment.save_csv(
        results,
        output_dir / "traffic_benchmark.csv",
    )

    print_results(results)

    print()
    print("Saved:")
    print("results/traffic_benchmark.json")
    print("results/traffic_benchmark.csv")
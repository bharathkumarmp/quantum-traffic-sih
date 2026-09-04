import time

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.data.scenario_graph_generator import ScenarioGraphGenerator
from src.optimization.fitness import FitnessFunction

class ScenarioBenchmark:
    """
    Benchmark Dijkstra, PSO and QPSO on controlled
    traffic scenarios.
    """

    def __init__(self, particles=20, iterations=50, seed=42):
        self.particles = particles
        self.iterations = iterations
        self.seed = seed

    def _algorithms(self, graph):
        fitness_function = FitnessFunction(
        time_weight=0.5,
        distance_weight=0.3,
        congestion_weight=0.2,
    )

        return {
        "Dijkstra": Dijkstra(
            graph,
            fitness_function=fitness_function,
        ),

        "PSO": PSO(
            graph,
            particles=self.particles,
            iterations=self.iterations,
            seed=self.seed,
            fitness_function=fitness_function,
            use_dijkstra_seed=True,
        ),

        "QPSO": QPSO(
            graph,
            particles=self.particles,
            iterations=self.iterations,
            seed=self.seed,
            fitness_function=fitness_function,
            use_dijkstra_seed=True,
        ),
    }

    def run(self):
        generator = ScenarioGraphGenerator()

        scenarios = {
            "normal": generator.generate(),
            "upper_route_incident": generator.upper_route_incident(),
            "upper_route_blockage": generator.upper_route_blockage(),
            "lower_route_incident": generator.lower_route_incident(),
        }

        results = []

        for scenario_name, graph in scenarios.items():
            algorithms = self._algorithms(graph)

            for algorithm_name, algorithm in algorithms.items():
                start_time = time.perf_counter()

                result = algorithm.optimize("N0", "N5")

                execution_time = time.perf_counter() - start_time

                route = result["route"]

                results.append(
                    {
                        "scenario": scenario_name,
                        "algorithm": algorithm_name,
                        "route": [node.id for node in route],
                        "fitness": result["fitness"],
                        "travel_time": result["travel_time"],
                        "distance": result["distance"],
                        "congestion": result["congestion"],
                        "execution_time": execution_time,
                        "iterations": result.get("iterations", 0),
                    }
                )

        return results

    def print_results(self):
        results = self.run()

        print("\n" + "=" * 90)
        print("CONTROLLED TRAFFIC ROUTE BENCHMARK")
        print("=" * 90)

        for result in results:
            print(
                f"{result['scenario']:25} "
                f"{result['algorithm']:10} "
                f"route={' -> '.join(result['route']):20} "
                f"fitness={result['fitness']:.4f} "
                f"time={result['travel_time']:.4f} "
                f"distance={result['distance']:.4f} "
                f"congestion={result['congestion']:.4f} "
                f"exec={result['execution_time']:.6f}s"
            )

        print("=" * 90)

        return results
if __name__ == "__main__":
    benchmark = ScenarioBenchmark()
    benchmark.print_results()
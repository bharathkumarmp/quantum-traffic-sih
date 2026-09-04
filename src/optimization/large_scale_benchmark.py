import csv
import time

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.data.graph_generator import LargeGraphGenerator
from src.optimization.fitness import FitnessFunction


class LargeScaleBenchmark:
    """
    Compare Dijkstra, PSO and QPSO on progressively
    larger transportation networks.
    """

    def __init__(
        self,
        network_sizes=(50, 100, 250, 500),
        particles=20,
        iterations=50,
        seed=42,
    ):
        self.network_sizes = network_sizes
        self.particles = particles
        self.iterations = iterations
        self.seed = seed

        self.fitness_function = FitnessFunction(
            time_weight=0.5,
            distance_weight=0.3,
            congestion_weight=0.2,
        )

    def _create_algorithms(self, graph):
        return {
            "Dijkstra": Dijkstra(
                graph,
                fitness_function=self.fitness_function,
            ),
            "PSO": PSO(
                graph,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
                use_dijkstra_seed=True,
            ),
            "QPSO": QPSO(
                graph,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
                use_dijkstra_seed=True,
            ),
        }

    def run(self):
        results = []

        for size in self.network_sizes:
            print(f"\nGenerating {size}-node network...")

            generator = LargeGraphGenerator(
                num_nodes=size,
                edges_per_node=3,
                seed=self.seed,
            )

            graph = generator.generate()

            start = f"N0"
            destination = f"N{size - 1}"

            algorithms = self._create_algorithms(graph)

            for algorithm_name, algorithm in algorithms.items():

                start_time = time.perf_counter()

                result = algorithm.optimize(
                    start,
                    destination,
                )

                execution_time = time.perf_counter() - start_time

                results.append(
                    {
                        "network_size": size,
                        "algorithm": algorithm_name,
                        "fitness": result["fitness"],
                        "travel_time": result["travel_time"],
                        "distance": result["distance"],
                        "congestion": result["congestion"],
                        "execution_time": execution_time,
                        "iterations": result.get("iterations", 0),
                    }
                )

                print(
                    f"{algorithm_name:10} | "
                    f"fitness={result['fitness']:.4f} | "
                    f"time={result['travel_time']:.4f} | "
                    f"distance={result['distance']:.4f} | "
                    f"congestion={result['congestion']:.4f} | "
                    f"execution={execution_time:.6f}s"
                )

        return results

    def save_csv(self, results, filename="results/large_scale_benchmark.csv"):
        if not results:
            return

        fieldnames = list(results[0].keys())

        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved results to: {filename}")

if __name__ == "__main__":
    benchmark = LargeScaleBenchmark()
    results = benchmark.run()
    benchmark.save_csv(results)        
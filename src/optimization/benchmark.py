import time
from statistics import mean


class BenchmarkResult:
    """Stores the result of one algorithm run."""

    def __init__(
        self,
        algorithm,
        fitness,
        travel_time,
        distance,
        congestion,
        execution_time,
        iterations,
    ):
        self.algorithm = algorithm
        self.fitness = float(fitness)
        self.travel_time = float(travel_time)
        self.distance = float(distance)
        self.congestion = float(congestion)
        self.execution_time = float(execution_time)
        self.iterations = int(iterations)

    def to_dict(self):
        return {
            "algorithm": self.algorithm,
            "fitness": self.fitness,
            "travel_time": self.travel_time,
            "distance": self.distance,
            "congestion": self.congestion,
            "execution_time": self.execution_time,
            "iterations": self.iterations,
        }


class Benchmark:
    """
    Benchmark optimization algorithms on the same graph.

    Every algorithm is evaluated between the same start
    and destination nodes so that the comparison is fair.
    """

    def __init__(
        self,
        graph,
        algorithms,
        start=None,
        destination=None,
    ):
        if graph is None:
            raise ValueError("Graph is required.")

        if not algorithms:
            raise ValueError("At least one algorithm is required.")

        self.graph = graph
        self.algorithms = algorithms

        nodes = self._get_graph_nodes()

        if not nodes:
            raise ValueError("Graph must contain nodes.")

        self.start = start if start is not None else nodes[0]
        self.destination = (
            destination if destination is not None else nodes[-1]
        )

        self.start = self._node_id(self.start)
        self.destination = self._node_id(self.destination)

    def _get_graph_nodes(self):
        """
        Retrieve graph nodes without requiring a new Graph API.
        """

        if hasattr(self.graph, "_node_map"):
            return list(self.graph._node_map.values())

        if hasattr(self.graph, "nodes"):
            nodes = self.graph.nodes
            if isinstance(nodes, dict):
                return list(nodes.values())
            return list(nodes)

        raise ValueError("Unable to retrieve graph nodes.")

    @staticmethod
    def _node_id(node):
        """Convert a Node object or node ID to its ID."""

        if hasattr(node, "id"):
            return node.id

        return str(node)

    def run(self):
        """Run all configured algorithms."""

        results = []

        for name, algorithm in self.algorithms.items():

            start_time = time.perf_counter()

            result = self._run_algorithm(algorithm)

            execution_time = time.perf_counter() - start_time

            results.append(
                BenchmarkResult(
                    algorithm=name,
                    fitness=result.get(
                        "fitness",
                        float("inf"),
                    ),
                    travel_time=result.get(
                        "travel_time",
                        float("inf"),
                    ),
                    distance=result.get(
                        "distance",
                        float("inf"),
                    ),
                    congestion=result.get(
                        "congestion",
                        float("inf"),
                    ),
                    execution_time=execution_time,
                    iterations=result.get(
                        "iterations",
                        0,
                    ),
                )
            )

        return results

    def _run_algorithm(self, algorithm):
        """Execute an optimization algorithm."""

        if not hasattr(algorithm, "optimize"):
            raise TypeError(
                "Algorithm must provide an optimize() method."
            )

        return algorithm.optimize(
            self.start,
            self.destination,
        )

    @staticmethod
    def summarize(results):
        """Convert benchmark results into dictionaries."""

        return [result.to_dict() for result in results]

    @staticmethod
    def best_by_fitness(results):
        """Return the result with the lowest fitness."""

        if not results:
            return None

        return min(
            results,
            key=lambda result: result.fitness,
        )

    @staticmethod
    def best_by_travel_time(results):
        """Return the result with the lowest travel time."""

        if not results:
            return None

        return min(
            results,
            key=lambda result: result.travel_time,
        )

    @staticmethod
    def average(results):
        """Calculate average benchmark metrics."""

        if not results:
            return {}

        return {
            "fitness": mean(
                result.fitness
                for result in results
            ),
            "travel_time": mean(
                result.travel_time
                for result in results
            ),
            "distance": mean(
                result.distance
                for result in results
            ),
            "congestion": mean(
                result.congestion
                for result in results
            ),
            "execution_time": mean(
                result.execution_time
                for result in results
            ),
        }
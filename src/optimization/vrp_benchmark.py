"""
VRP benchmark for comparing classical PSO and quantum-inspired QPSO.

The benchmark evaluates both algorithms on the same:
- graph
- customers
- vehicles
- traffic conditions
- fitness function
- random seed

This is intended for experimental comparison, not for claiming
that QPSO is always superior to PSO.
"""

import csv
import json
import os
import time

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.data.graph_generator import LargeGraphGenerator
from src.domain.customer import Customer
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem
from src.optimization.vrp_fitness import VRPFitnessFunction


class VRPBenchmark:
    """
    Benchmark PSO and QPSO on progressively larger VRP instances.
    """

    def __init__(
        self,
        customer_sizes=(10, 20, 50, 100),
        particles=20,
        iterations=50,
        seed=42,
    ):
        self.customer_sizes = tuple(customer_sizes)
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.seed = seed

        self.fitness_function = VRPFitnessFunction(
            time_weight=0.5,
            distance_weight=0.3,
            congestion_weight=0.2,
        )

    def _build_problem(self, num_customers):
        """
        Build a deterministic VRP instance.

        The graph contains the depot and customer nodes.
        Customer demand and vehicle capacities are generated
        deterministically from the benchmark size.
        """

        num_nodes = max(num_customers + 1, 20)

        graph = LargeGraphGenerator(
            num_nodes=num_nodes,
            edges_per_node=4,
            seed=self.seed,
        ).generate()

        depot = "N0"

        customers = []

        for i in range(1, num_customers + 1):
            customers.append(
                Customer(
                    id=f"C{i}",
                    demand=1.0 + (i % 4),
                    node_id=f"N{i}",
                )
            )

        total_demand = sum(
            customer.demand
            for customer in customers
        )

        # Five vehicles provide enough total capacity
        # while still creating meaningful assignment constraints.
        vehicle_count = max(
            2,
            min(10, (num_customers + 9) // 10),
        )

        capacity_per_vehicle = (
            total_demand / vehicle_count
        ) * 1.25

        vehicles = []

        for i in range(vehicle_count):
            vehicles.append(
                Vehicle(
                    id=f"V{i + 1}",
                    capacity=capacity_per_vehicle,
                    start_node=depot,
                    end_node=depot,
                )
            )

        problem = VRPProblem(
            graph=graph,
            depot=depot,
            customers=customers,
            vehicles=vehicles,
        )

        if not problem.is_capacity_feasible():
            raise RuntimeError(
                "Generated VRP instance is capacity infeasible."
            )

        return problem

    def _run_algorithm(
        self,
        algorithm_name,
        problem,
    ):
        """
        Run one algorithm and measure execution time.
        """

        if algorithm_name == "PSO":
            algorithm = VRPPSO(
                problem=problem,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
            )

        elif algorithm_name == "QPSO":
            algorithm = VRPQPSO(
                problem=problem,
                particles=self.particles,
                iterations=self.iterations,
                seed=self.seed,
                fitness_function=self.fitness_function,
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}"
            )

        start_time = time.perf_counter()

        result = algorithm.optimize()

        execution_time = (
            time.perf_counter() - start_time
        )

        history = result.get("history", [])

        return {
            "algorithm": algorithm_name,
            "customers": problem.number_of_customers,
            "vehicles": problem.number_of_vehicles,
            "fitness": result["fitness"],
            "travel_time": result["travel_time"],
            "distance": result["distance"],
            "congestion": result["congestion"],
            "execution_time": execution_time,
            "iterations": result["iterations"],
            "history": history,
            "valid": bool(result.get("solution") is not None),
        }

    def run(self):
        """
        Run the complete benchmark.
        """

        results = []

        for customer_count in self.customer_sizes:

            print(
                f"\nRunning VRP benchmark "
                f"for {customer_count} customers..."
            )

            problem = self._build_problem(
                customer_count
            )

            for algorithm_name in ("PSO", "QPSO"):

                result = self._run_algorithm(
                    algorithm_name,
                    problem,
                )

                results.append(result)

                print(
                    f"{algorithm_name}: "
                    f"fitness={result['fitness']:.4f}, "
                    f"time={result['travel_time']:.4f}, "
                    f"distance={result['distance']:.4f}, "
                    f"execution={result['execution_time']:.4f}s"
                )

        return results

    @staticmethod
    def summarize(results):
        """
        Create a compact summary suitable for tables/charts.
        """

        summary = []

        for result in results:

            summary.append(
                {
                    "algorithm": result["algorithm"],
                    "customers": result["customers"],
                    "vehicles": result["vehicles"],
                    "fitness": result["fitness"],
                    "travel_time": result["travel_time"],
                    "distance": result["distance"],
                    "congestion": result["congestion"],
                    "execution_time": result["execution_time"],
                    "iterations": result["iterations"],
                    "valid": result["valid"],
                }
            )

        return summary

    @staticmethod
    def save_json(results, path):
        os.makedirs(
            os.path.dirname(path) or ".",
            exist_ok=True,
        )

        with open(
            path,
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
        os.makedirs(
            os.path.dirname(path) or ".",
            exist_ok=True,
        )

        rows = VRPBenchmark.summarize(results)

        if not rows:
            return

        fieldnames = list(rows[0].keys())

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)


def main():
    benchmark = VRPBenchmark(
        customer_sizes=(10, 20, 50, 100),
        particles=20,
        iterations=50,
        seed=42,
    )

    results = benchmark.run()

    benchmark.save_json(
        results,
        "results/vrp_benchmark.json",
    )

    benchmark.save_csv(
        results,
        "results/vrp_benchmark.csv",
    )

    print(
        "\nVRP benchmark completed."
    )

    print(
        "Saved:"
    )
    print(
        "  results/vrp_benchmark.json"
    )
    print(
        "  results/vrp_benchmark.csv"
    )


if __name__ == "__main__":
    main()
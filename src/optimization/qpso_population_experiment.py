"""
QPSO population-size experiment.

Tests how the number of particles affects QPSO performance
on a fixed large-scale VRP instance.

Controlled variables:
    customers  = 100
    iterations  = 50
    beta        = 0.7
    seed        = 42

Independent variable:
    particles = 10, 20, 30, 50
"""

import csv
import os
import time

from src.algorithms.vrp_qpso import VRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark


def main():

    population_sizes = (
        10,
        20,
        30,
        50,
    )

    benchmark = VRPBenchmark(
        customer_sizes=(100,),
        particles=20,
        iterations=50,
        seed=42,
    )

    problem = benchmark._build_problem(100)

    results = []

    print("\nQPSO population-size experiment")
    print("--------------------------------")
    print("Customers : 100")
    print("Iterations: 50")
    print("Beta      : 0.7")
    print("Seed      : 42\n")

    for particles in population_sizes:

        algorithm = VRPQPSO(
            problem=problem,
            particles=particles,
            iterations=50,
            beta=0.7,
            seed=42,
            fitness_function=benchmark.fitness_function,
        )

        start = time.perf_counter()

        result = algorithm.optimize()

        execution_time = (
            time.perf_counter() - start
        )

        history = result.get("history", [])

        initial_fitness = (
            history[0]
            if history
            else result["fitness"]
        )

        final_fitness = result["fitness"]

        improvement_percent = 0.0

        if initial_fitness > 0:
            improvement_percent = (
                (initial_fitness - final_fitness)
                / initial_fitness
            ) * 100.0

        row = {
            "particles": particles,
            "fitness": final_fitness,
            "travel_time": result["travel_time"],
            "distance": result["distance"],
            "congestion": result["congestion"],
            "execution_time": execution_time,
            "initial_fitness": initial_fitness,
            "improvement_percent": improvement_percent,
            "iterations": result["iterations"],
        }

        results.append(row)

        print(
            f"particles={particles:2d} | "
            f"fitness={final_fitness:.4f} | "
            f"time={result['travel_time']:.4f} | "
            f"distance={result['distance']:.4f} | "
            f"improvement={improvement_percent:.2f}% | "
            f"execution={execution_time:.4f}s"
        )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    output_path = (
        "results/qpso_population_experiment.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
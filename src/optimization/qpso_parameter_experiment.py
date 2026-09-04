import csv
import os
import time

from src.algorithms.vrp_qpso import VRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark


def main():

    beta_values = (
        0.3,
        0.5,
        0.7,
        0.9,
        1.0,
    )

    benchmark = VRPBenchmark(
        customer_sizes=(100,),
        particles=20,
        iterations=50,
        seed=42,
    )

    problem = benchmark._build_problem(100)

    results = []

    print("\nQPSO beta parameter experiment")
    print("--------------------------------")

    for beta in beta_values:

        algorithm = VRPQPSO(
            problem=problem,
            particles=20,
            iterations=50,
            beta=beta,
            seed=42,
            fitness_function=benchmark.fitness_function,
        )

        start = time.perf_counter()

        result = algorithm.optimize()

        execution_time = (
            time.perf_counter() - start
        )

        history = result.get("history", [])

        row = {
            "beta": beta,
            "fitness": result["fitness"],
            "travel_time": result["travel_time"],
            "distance": result["distance"],
            "congestion": result["congestion"],
            "execution_time": execution_time,
            "initial_fitness": history[0],
            "final_fitness": history[-1],
        }

        results.append(row)

        print(
            f"beta={beta:.1f} | "
            f"fitness={result['fitness']:.4f} | "
            f"time={result['travel_time']:.4f} | "
            f"distance={result['distance']:.4f} | "
            f"execution={execution_time:.4f}s"
        )

    os.makedirs("results", exist_ok=True)

    output = (
        "results/qpso_parameter_experiment.csv"
    )

    with open(
        output,
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
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()
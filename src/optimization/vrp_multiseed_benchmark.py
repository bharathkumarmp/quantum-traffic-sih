"""
Multi-seed PSO vs QPSO VRP experiment.

Purpose:
    Evaluate whether algorithm performance is robust
    across different random initializations.

Controlled:
    same VRP problem
    same fitness function
    same particles
    same iterations
    same beta

Variable:
    random seed
"""

import csv
import os
import statistics
import time

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark


def run_algorithm(
    algorithm_name,
    problem,
    fitness_function,
    seed,
):
    if algorithm_name == "PSO":
        algorithm = VRPPSO(
            problem=problem,
            particles=20,
            iterations=50,
            seed=seed,
            fitness_function=fitness_function,
        )

    elif algorithm_name == "QPSO":
        algorithm = VRPQPSO(
            problem=problem,
            particles=20,
            iterations=50,
            beta=0.7,
            seed=seed,
            fitness_function=fitness_function,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )

    start = time.perf_counter()

    result = algorithm.optimize()

    execution_time = (
        time.perf_counter() - start
    )

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "execution_time": execution_time,
    }


def main():

    customer_sizes = (
        10,
        20,
        50,
    )

    seeds = (
        42,
        7,
        21,
        100,
        123,
    )

    benchmark = VRPBenchmark(
        customer_sizes=(10,),
        particles=20,
        iterations=50,
        seed=42,
    )

    all_results = []

    print("\nMulti-seed VRP benchmark")
    print("========================")

    for customer_count in customer_sizes:

        print(
            f"\n--- {customer_count} customers ---"
        )

        # Same graph/problem structure for all algorithms
        # within this customer size.
        problem = benchmark._build_problem(
            customer_count
        )

        for seed in seeds:

            for algorithm_name in (
                "PSO",
                "QPSO",
            ):

                result = run_algorithm(
                    algorithm_name,
                    problem,
                    benchmark.fitness_function,
                    seed,
                )

                result["customers"] = (
                    customer_count
                )

                all_results.append(result)

                print(
                    f"{algorithm_name:5s} | "
                    f"seed={seed:3d} | "
                    f"fitness={result['fitness']:.4f} | "
                    f"time={result['execution_time']:.4f}s"
                )

    # ------------------------------------------------------------
    # Save individual runs
    # ------------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    raw_path = (
        "results/vrp_multiseed_results.csv"
    )

    with open(
        raw_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "customers",
            "algorithm",
            "seed",
            "fitness",
            "travel_time",
            "distance",
            "congestion",
            "execution_time",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    # ------------------------------------------------------------
    # Calculate summary statistics
    # ------------------------------------------------------------

    summary = []

    print("\nSummary")
    print("=======")

    for customer_count in customer_sizes:

        for algorithm_name in (
            "PSO",
            "QPSO",
        ):

            matching = [
                result
                for result in all_results
                if (
                    result["customers"]
                    == customer_count
                    and result["algorithm"]
                    == algorithm_name
                )
            ]

            fitness_values = [
                result["fitness"]
                for result in matching
            ]

            execution_values = [
                result["execution_time"]
                for result in matching
            ]

            mean_fitness = (
                statistics.mean(
                    fitness_values
                )
            )

            std_fitness = (
                statistics.stdev(
                    fitness_values
                )
                if len(fitness_values) > 1
                else 0.0
            )

            best_fitness = min(
                fitness_values
            )

            mean_execution = (
                statistics.mean(
                    execution_values
                )
            )

            row = {
                "customers": customer_count,
                "algorithm": algorithm_name,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "best_fitness": best_fitness,
                "mean_execution_time": mean_execution,
            }

            summary.append(row)

            print(
                f"{algorithm_name:5s} | "
                f"mean={mean_fitness:.4f} | "
                f"std={std_fitness:.4f} | "
                f"best={best_fitness:.4f} | "
                f"mean_time={mean_execution:.4f}s"
            )

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------

    summary_path = (
        "results/vrp_multiseed_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "customers",
            "algorithm",
            "mean_fitness",
            "std_fitness",
            "best_fitness",
            "mean_execution_time",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nSaved:")
    print(
        f"  {raw_path}"
    )
    print(
        f"  {summary_path}"
    )


if __name__ == "__main__":
    main()
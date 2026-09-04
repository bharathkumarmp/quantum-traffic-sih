import csv
import os
import time

from src.algorithms.decomposed_warm_qpso import (
    DecomposedWarmQPSO,
)
from src.optimization.vrp_benchmark import VRPBenchmark
from src.optimization.vrp_fitness import VRPFitnessFunction


def run_experiment(
    problem,
    clusters,
    seed,
    fitness_function,
):
    """Run decomposition with a specific number of clusters."""

    print(
        f"\nRunning {clusters} clusters "
        f"({problem.number_of_customers} customers)..."
    )

    optimizer = DecomposedWarmQPSO(
        problem=problem,
        clusters=clusters,
        particles=20,
        iterations=50,
        beta=0.7,
        seed=seed,
        fitness_function=fitness_function,
        warm_iterations=10,
        warm_particles=10,
    )

    start = time.perf_counter()

    result = optimizer.optimize()

    runtime = time.perf_counter() - start

    print(
        f"Clusters:      {clusters}"
    )

    print(
        f"Customers/cluster: "
        f"~{problem.number_of_customers / clusters:.1f}"
    )

    print(
        f"Fitness:        "
        f"{result['fitness']:.4f}"
    )

    print(
        f"Travel time:    "
        f"{result['travel_time']:.4f}"
    )

    print(
        f"Distance:       "
        f"{result['distance']:.4f}"
    )

    print(
        f"Congestion:     "
        f"{result['congestion']:.4f}"
    )

    print(
        f"Runtime:        "
        f"{runtime:.2f}s"
    )

    return {
        "customers": problem.number_of_customers,
        "clusters": clusters,
        "customers_per_cluster": (
            problem.number_of_customers / clusters
        ),
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
    }


def main():

    # =========================================================
    # CONFIGURATION
    # =========================================================

    customer_count = 100
    seed = 42

    cluster_counts = (
        2,
        4,
        5,
        10,
    )

    print("\n")
    print("=" * 100)
    print("DECOMPOSITION SCALING EXPERIMENT")
    print("=" * 100)

    print(
        f"Customers : {customer_count}"
    )

    print(
        f"Clusters  : {cluster_counts}"
    )

    print(
        f"Particles : 20"
    )

    print(
        f"Iterations: 50"
    )

    print(
        f"Seed      : {seed}"
    )

    print(
        f"Warm PSO  : 10 particles × 10 iterations"
    )

    print("=" * 100)

    # =========================================================
    # BUILD ONE FIXED PROBLEM
    # =========================================================

    benchmark = VRPBenchmark(
        customer_sizes=(customer_count,),
        particles=20,
        iterations=50,
        seed=seed,
    )

    problem = benchmark._build_problem(
        customer_count
    )

    fitness_function = VRPFitnessFunction(
        time_weight=0.5,
        distance_weight=0.3,
        congestion_weight=0.2,
    )

    results = []

    # =========================================================
    # RUN EXPERIMENT
    # =========================================================

    total_start = time.perf_counter()

    for clusters in cluster_counts:

        result = run_experiment(
            problem=problem,
            clusters=clusters,
            seed=seed,
            fitness_function=fitness_function,
        )

        results.append(result)

    total_runtime = (
        time.perf_counter() - total_start
    )

    # =========================================================
    # FINAL TABLE
    # =========================================================

    print("\n")
    print("=" * 115)
    print("DECOMPOSITION SCALING RESULTS")
    print("=" * 115)

    print(
        f"{'Clusters':<12}"
        f"{'Cust/Cluster':>16}"
        f"{'Fitness':>14}"
        f"{'Travel Time':>16}"
        f"{'Distance':>14}"
        f"{'Congestion':>14}"
        f"{'Runtime':>14}"
    )

    print("-" * 115)

    for result in results:

        print(
            f"{result['clusters']:<12}"
            f"{result['customers_per_cluster']:>16.1f}"
            f"{result['fitness']:>14.4f}"
            f"{result['travel_time']:>16.4f}"
            f"{result['distance']:>14.4f}"
            f"{result['congestion']:>14.4f}"
            f"{result['runtime']:>13.2f}s"
        )

    # =========================================================
    # BEST CLUSTER COUNT
    # =========================================================

    best = min(
        results,
        key=lambda x: x["fitness"],
    )

    print("-" * 115)

    print(
        f"\nBEST DECOMPOSITION:"
    )

    print(
        f"{best['clusters']} clusters "
        f"({best['customers_per_cluster']:.1f} "
        f"customers/cluster)"
    )

    print(
        f"Fitness = "
        f"{best['fitness']:.4f}"
    )

    print(
        f"Runtime = "
        f"{best['runtime']:.2f}s"
    )

    # =========================================================
    # SAVE CSV
    # =========================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )

    output_file = (
        "results/"
        "decomposition_scaling_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "customers",
            "clusters",
            "customers_per_cluster",
            "fitness",
            "travel_time",
            "distance",
            "congestion",
            "runtime",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nResults saved to: "
        f"{output_file}"
    )

    print(
        f"Total experiment runtime: "
        f"{total_runtime:.2f}s"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
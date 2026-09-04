import csv
import os
import statistics
import time

from src.algorithms.decomposed_warm_qpso import DecomposedWarmQPSO
from src.optimization.vrp_benchmark import VRPBenchmark
from src.optimization.vrp_fitness import VRPFitnessFunction


CUSTOMERS = 100
CLUSTER_COUNTS = (2, 4, 5, 10)
SEEDS = (42, 7, 21, 100, 123)

PARTICLES = 20
ITERATIONS = 50
BETA = 0.7

WARM_PARTICLES = 10
WARM_ITERATIONS = 10


def build_problem():
    benchmark = VRPBenchmark(
        customer_sizes=(CUSTOMERS,),
        particles=PARTICLES,
        iterations=ITERATIONS,
        seed=42,
    )
    return benchmark._build_problem(CUSTOMERS)


def run_single(problem, clusters, seed, fitness_function):
    print(
        f"  Seed {seed:>3} | "
        f"{clusters:>2} clusters | ",
        end="",
        flush=True,
    )

    optimizer = DecomposedWarmQPSO(
        problem=problem,
        clusters=clusters,
        particles=PARTICLES,
        iterations=ITERATIONS,
        beta=BETA,
        seed=seed,
        fitness_function=fitness_function,
        warm_iterations=WARM_ITERATIONS,
        warm_particles=WARM_PARTICLES,
    )

    start = time.perf_counter()
    result = optimizer.optimize()
    runtime = time.perf_counter() - start

    print(
        f"fitness={result['fitness']:.4f} | "
        f"runtime={runtime:.2f}s"
    )

    return {
        "customers": CUSTOMERS,
        "clusters": clusters,
        "seed": seed,
        "customers_per_cluster": CUSTOMERS / clusters,
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "runtime": runtime,
    }


def summarize(results):
    summaries = []

    for clusters in CLUSTER_COUNTS:
        rows = [
            row for row in results
            if row["clusters"] == clusters
        ]

        fitness_values = [row["fitness"] for row in rows]
        travel_values = [row["travel_time"] for row in rows]
        distance_values = [row["distance"] for row in rows]
        congestion_values = [row["congestion"] for row in rows]
        runtime_values = [row["runtime"] for row in rows]

        summaries.append({
            "clusters": clusters,
            "customers_per_cluster": CUSTOMERS / clusters,
            "mean_fitness": statistics.mean(fitness_values),
            "std_fitness": statistics.stdev(fitness_values),
            "best_fitness": min(fitness_values),
            "worst_fitness": max(fitness_values),
            "mean_travel_time": statistics.mean(travel_values),
            "mean_distance": statistics.mean(distance_values),
            "mean_congestion": statistics.mean(congestion_values),
            "mean_runtime": statistics.mean(runtime_values),
        })

    return summaries


def main():
    print("\n")
    print("=" * 100)
    print("DECOMPOSED WARM-QPSO MULTI-SEED EXPERIMENT")
    print("=" * 100)
    print(f"Customers       : {CUSTOMERS}")
    print(f"Clusters        : {CLUSTER_COUNTS}")
    print(f"Seeds           : {SEEDS}")
    print(f"Particles       : {PARTICLES}")
    print(f"Iterations      : {ITERATIONS}")
    print(f"Beta            : {BETA}")
    print(
        f"Warm-QPSO       : "
        f"{WARM_PARTICLES} particles × "
        f"{WARM_ITERATIONS} iterations"
    )
    print("=" * 100)

    problem = build_problem()

    fitness_function = VRPFitnessFunction(
        time_weight=0.5,
        distance_weight=0.3,
        congestion_weight=0.2,
    )

    results = []

    experiment_start = time.perf_counter()

    for clusters in CLUSTER_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"CLUSTER COUNT: {clusters}")
        print(f"{'=' * 70}")

        for seed in SEEDS:
            result = run_single(
                problem=problem,
                clusters=clusters,
                seed=seed,
                fitness_function=fitness_function,
            )
            results.append(result)

    total_runtime = time.perf_counter() - experiment_start

    summaries = summarize(results)

    print("\n")
    print("=" * 125)
    print("MULTI-SEED DECOMPOSITION RESULTS")
    print("=" * 125)

    print(
        f"{'Clusters':<10}"
        f"{'Cust/Cluster':>15}"
        f"{'Mean Fitness':>16}"
        f"{'Std Dev':>14}"
        f"{'Best':>14}"
        f"{'Worst':>14}"
        f"{'Mean Runtime':>16}"
    )

    print("-" * 125)

    for row in summaries:
        print(
            f"{row['clusters']:<10}"
            f"{row['customers_per_cluster']:>15.1f}"
            f"{row['mean_fitness']:>16.4f}"
            f"{row['std_fitness']:>14.4f}"
            f"{row['best_fitness']:>14.4f}"
            f"{row['worst_fitness']:>14.4f}"
            f"{row['mean_runtime']:>14.2f}s"
        )

    best = min(
        summaries,
        key=lambda row: row["mean_fitness"]
    )

    print("-" * 125)

    print("\nBEST CLUSTER CONFIGURATION")
    print(
        f"{best['clusters']} clusters "
        f"({best['customers_per_cluster']:.1f} customers/cluster)"
    )
    print(f"Mean fitness : {best['mean_fitness']:.4f}")
    print(f"Std deviation: {best['std_fitness']:.4f}")
    print(f"Best fitness : {best['best_fitness']:.4f}")
    print(f"Worst fitness: {best['worst_fitness']:.4f}")
    print(f"Mean runtime : {best['mean_runtime']:.2f}s")

    os.makedirs("results", exist_ok=True)

    raw_file = "results/decomposition_multiseed_results.csv"

    with open(raw_file, "w", newline="") as file:
        fieldnames = [
            "customers",
            "clusters",
            "seed",
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

    summary_file = (
        "results/decomposition_multiseed_summary.csv"
    )

    with open(summary_file, "w", newline="") as file:
        fieldnames = [
            "clusters",
            "customers_per_cluster",
            "mean_fitness",
            "std_fitness",
            "best_fitness",
            "worst_fitness",
            "mean_travel_time",
            "mean_distance",
            "mean_congestion",
            "mean_runtime",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summaries)

    print("\nResults saved:")
    print(f"  {raw_file}")
    print(f"  {summary_file}")

    print(f"\nTotal experiment runtime: {total_runtime:.2f}s")

    print("=" * 100)


if __name__ == "__main__":
    main()
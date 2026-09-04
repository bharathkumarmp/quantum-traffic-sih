import csv
import os
import statistics
import time

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark
from src.optimization.vrp_fitness import VRPFitnessFunction


CUSTOMER_SIZES = (10, 20, 50, 100)
SEEDS = (42, 7, 21, 100, 123)

PARTICLES = 20
ITERATIONS = 50

WARM_PARTICLES = 10
WARM_ITERATIONS = 10

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "warm_vrp_multiseed_results.csv",
)


def create_algorithms(
    problem,
    seed,
    fitness_function,
):
    return [
        (
            "PSO",
            VRPPSO(
                problem=problem,
                particles=PARTICLES,
                iterations=ITERATIONS,
                seed=seed,
                fitness_function=fitness_function,
            ),
        ),
        (
            "QPSO",
            VRPQPSO(
                problem=problem,
                particles=PARTICLES,
                iterations=ITERATIONS,
                beta=0.7,
                seed=seed,
                fitness_function=fitness_function,
            ),
        ),
        (
            "Warm-QPSO",
            WarmVRPQPSO(
                problem=problem,
                particles=PARTICLES,
                iterations=ITERATIONS,
                beta=0.7,
                seed=seed,
                fitness_function=fitness_function,
                warm_iterations=WARM_ITERATIONS,
                warm_particles=WARM_PARTICLES,
            ),
        ),
    ]


def run_algorithm(name, optimizer):
    start = time.perf_counter()

    result = optimizer.optimize()

    execution_time = time.perf_counter() - start

    return {
        "algorithm": name,
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "execution_time": execution_time,
        "warm_start_fitness": result.get("warm_start_fitness"),
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    print("\n")
    print("=" * 100)
    print("MULTI-SEED PSO vs QPSO vs WARM-QPSO BENCHMARK")
    print("=" * 100)

    print(f"Customer sizes : {CUSTOMER_SIZES}")
    print(f"Seeds          : {SEEDS}")
    print(f"Particles      : {PARTICLES}")
    print(f"Iterations     : {ITERATIONS}")
    print(f"Warm particles : {WARM_PARTICLES}")
    print(f"Warm iterations: {WARM_ITERATIONS}")
    print("=" * 100)

    total_start = time.perf_counter()

    for customer_count in CUSTOMER_SIZES:

        for seed in SEEDS:

            print("\n" + "-" * 100)
            print(
                f"Customers = {customer_count} | "
                f"Seed = {seed}"
            )
            print("-" * 100)

            benchmark = VRPBenchmark(
                customer_sizes=(customer_count,),
                particles=PARTICLES,
                iterations=ITERATIONS,
                seed=seed,
            )

            problem = benchmark._build_problem(customer_count)

            fitness_function = VRPFitnessFunction(
                time_weight=0.5,
                distance_weight=0.3,
                congestion_weight=0.2,
            )

            algorithms = create_algorithms(
                problem=problem,
                seed=seed,
                fitness_function=fitness_function,
            )

            for name, optimizer in algorithms:

                print(f"Running {name}...", end=" ", flush=True)

                result = run_algorithm(name, optimizer)

                result["customer_count"] = customer_count
                result["seed"] = seed

                all_results.append(result)

                print(
                    f"fitness={result['fitness']:.4f}, "
                    f"time={result['execution_time']:.2f}s"
                )

    total_time = time.perf_counter() - total_start

    # ---------------------------------------------------------
    # SAVE RAW RESULTS
    # ---------------------------------------------------------

    fieldnames = [
        "customer_count",
        "seed",
        "algorithm",
        "fitness",
        "travel_time",
        "distance",
        "congestion",
        "execution_time",
        "warm_start_fitness",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n\n")
    print("=" * 110)
    print("MULTI-SEED SUMMARY")
    print("=" * 110)

    print(
        f"{'Customers':<12}"
        f"{'Algorithm':<15}"
        f"{'Mean Fitness':>15}"
        f"{'Std Dev':>12}"
        f"{'Best':>14}"
        f"{'Worst':>14}"
        f"{'Mean Runtime':>16}"
    )

    print("-" * 110)

    summary = []

    for customer_count in CUSTOMER_SIZES:

        for algorithm in (
            "PSO",
            "QPSO",
            "Warm-QPSO",
        ):

            rows = [
                r
                for r in all_results
                if r["customer_count"] == customer_count
                and r["algorithm"] == algorithm
            ]

            fitness_values = [
                r["fitness"]
                for r in rows
            ]

            runtime_values = [
                r["execution_time"]
                for r in rows
            ]

            mean_fitness = statistics.mean(
                fitness_values
            )

            std_fitness = statistics.stdev(
                fitness_values
            )

            best_fitness = min(
                fitness_values
            )

            worst_fitness = max(
                fitness_values
            )

            mean_runtime = statistics.mean(
                runtime_values
            )

            summary_row = {
                "customer_count": customer_count,
                "algorithm": algorithm,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "best_fitness": best_fitness,
                "worst_fitness": worst_fitness,
                "mean_runtime": mean_runtime,
            }

            summary.append(summary_row)

            print(
                f"{customer_count:<12}"
                f"{algorithm:<15}"
                f"{mean_fitness:>15.4f}"
                f"{std_fitness:>12.4f}"
                f"{best_fitness:>14.4f}"
                f"{worst_fitness:>14.4f}"
                f"{mean_runtime:>15.4f}s"
            )

        print("-" * 110)

    # ---------------------------------------------------------
    # WIN COUNT
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("WIN COUNT BY CUSTOMER SIZE")
    print("=" * 80)

    win_counts = {
        "PSO": 0,
        "QPSO": 0,
        "Warm-QPSO": 0,
    }

    for customer_count in CUSTOMER_SIZES:

        customer_summary = [
            row
            for row in summary
            if row["customer_count"] == customer_count
        ]

        winner = min(
            customer_summary,
            key=lambda x: x["mean_fitness"],
        )

        win_counts[winner["algorithm"]] += 1

        print(
            f"{customer_count:>3} customers → "
            f"{winner['algorithm']:<12} "
            f"mean fitness = "
            f"{winner['mean_fitness']:.4f}"
        )

    print("\nOverall mean-fitness wins:")

    for algorithm, count in win_counts.items():
        print(f"{algorithm:<12}: {count}")

    # ---------------------------------------------------------
    # WARM-QPSO IMPROVEMENT
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("WARM-QPSO WARM-START IMPROVEMENT")
    print("=" * 80)

    for customer_count in CUSTOMER_SIZES:

        rows = [
            r
            for r in all_results
            if r["customer_count"] == customer_count
            and r["algorithm"] == "Warm-QPSO"
        ]

        improvements = []

        for row in rows:

            warm_start = row["warm_start_fitness"]
            final_fitness = row["fitness"]

            if warm_start is not None and warm_start > 0:

                improvement = (
                    (warm_start - final_fitness)
                    / warm_start
                    * 100
                )

                improvements.append(improvement)

        if improvements:

            print(
                f"{customer_count:>3} customers → "
                f"average improvement = "
                f"{statistics.mean(improvements):.2f}%"
            )

    # ---------------------------------------------------------
    # FINAL INFORMATION
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)

    print(
        f"Total runs     : {len(all_results)}"
    )

    print(
        f"Total runtime  : {total_time:.2f}s"
    )

    print(
        f"Results saved  : {OUTPUT_FILE}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
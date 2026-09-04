import time

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark
from src.optimization.vrp_fitness import VRPFitnessFunction


def run_algorithm(name, optimizer):
    """Run an optimizer and collect benchmark metrics."""

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


def create_algorithms(
    problem,
    particles,
    iterations,
    seed,
    fitness_function,
):
    """Create PSO, QPSO and Warm-QPSO with identical main settings."""

    return [
        (
            "PSO",
            VRPPSO(
                problem=problem,
                particles=particles,
                iterations=iterations,
                seed=seed,
                fitness_function=fitness_function,
            ),
        ),
        (
            "QPSO",
            VRPQPSO(
                problem=problem,
                particles=particles,
                iterations=iterations,
                beta=0.7,
                seed=seed,
                fitness_function=fitness_function,
            ),
        ),
        (
            "Warm-QPSO",
            WarmVRPQPSO(
                problem=problem,
                particles=particles,
                iterations=iterations,
                beta=0.7,
                seed=seed,
                fitness_function=fitness_function,
                warm_iterations=10,
                warm_particles=10,
            ),
        ),
    ]


def run_customer_size(
    customer_count,
    seed=42,
    particles=20,
    iterations=50,
):
    """Run all algorithms for one customer size."""

    print("\n")
    print("=" * 100)
    print(f"BENCHMARK: {customer_count} CUSTOMERS")
    print("=" * 100)

    benchmark = VRPBenchmark(
        customer_sizes=(customer_count,),
        particles=particles,
        iterations=iterations,
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
        particles=particles,
        iterations=iterations,
        seed=seed,
        fitness_function=fitness_function,
    )

    results = []

    for name, optimizer in algorithms:

        print(f"\nRunning {name}...")

        result = run_algorithm(name, optimizer)

        results.append(result)

        print(f"Fitness:       {result['fitness']:.4f}")
        print(f"Travel time:   {result['travel_time']:.4f}")
        print(f"Distance:      {result['distance']:.4f}")
        print(f"Congestion:    {result['congestion']:.4f}")
        print(f"Execution:     {result['execution_time']:.4f}s")

        if result["warm_start_fitness"] is not None:
            print(
                f"Warm-start:    "
                f"{result['warm_start_fitness']:.4f}"
            )

    print("\n" + "-" * 100)
    print(f"COMPARISON — {customer_count} CUSTOMERS")
    print("-" * 100)

    print(
        f"{'Algorithm':<15}"
        f"{'Fitness':>14}"
        f"{'Travel Time':>16}"
        f"{'Distance':>14}"
        f"{'Congestion':>14}"
        f"{'Runtime':>14}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['algorithm']:<15}"
            f"{result['fitness']:>14.4f}"
            f"{result['travel_time']:>16.4f}"
            f"{result['distance']:>14.4f}"
            f"{result['congestion']:>14.4f}"
            f"{result['execution_time']:>13.4f}s"
        )

    best = min(results, key=lambda x: x["fitness"])

    print("-" * 100)

    print(
        f"BEST: {best['algorithm']} "
        f"(Fitness = {best['fitness']:.4f})"
    )

    return results


def print_overall_summary(all_results):
    """Print final comparison for all customer sizes."""

    print("\n\n")
    print("=" * 115)
    print("OVERALL PSO vs QPSO vs WARM-QPSO BENCHMARK")
    print("=" * 115)

    print(
        f"{'Customers':<12}"
        f"{'Algorithm':<15}"
        f"{'Fitness':>14}"
        f"{'Travel Time':>16}"
        f"{'Distance':>14}"
        f"{'Congestion':>14}"
        f"{'Runtime':>14}"
    )

    print("-" * 115)

    for customer_count, results in all_results.items():

        for result in results:

            print(
                f"{customer_count:<12}"
                f"{result['algorithm']:<15}"
                f"{result['fitness']:>14.4f}"
                f"{result['travel_time']:>16.4f}"
                f"{result['distance']:>14.4f}"
                f"{result['congestion']:>14.4f}"
                f"{result['execution_time']:>13.4f}s"
            )

        print("-" * 115)

    print("\nBEST ALGORITHM BY CUSTOMER SIZE")
    print("-" * 60)

    for customer_count, results in all_results.items():

        best = min(results, key=lambda x: x["fitness"])

        print(
            f"{customer_count:>3} customers → "
            f"{best['algorithm']:<12} "
            f"Fitness = {best['fitness']:.4f}"
        )

    print("\n")


def main():

    # =========================================================
    # EXPERIMENT CONFIGURATION
    # =========================================================

    seed = 42

    # Now includes 100 customers
    customer_sizes = (10, 20, 50, 100)

    particles = 20
    iterations = 50

    # Warm-QPSO configuration
    warm_particles = 10
    warm_iterations = 10

    print("\n")
    print("#" * 100)
    print("#")
    print("#        QUANTUM-INSPIRED VRP OPTIMIZATION BENCHMARK")
    print("#        PSO vs QPSO vs WARM-QPSO")
    print("#")
    print("#" * 100)

    print("\nExperiment configuration:")
    print(f"Customer sizes : {customer_sizes}")
    print(f"Particles      : {particles}")
    print(f"Iterations     : {iterations}")
    print(f"Seed           : {seed}")
    print(f"Warm particles : {warm_particles}")
    print(f"Warm iterations: {warm_iterations}")

    all_results = {}

    # =========================================================
    # RUN ALL CUSTOMER SIZES
    # =========================================================

    for customer_count in customer_sizes:

        results = run_customer_size(
            customer_count=customer_count,
            seed=seed,
            particles=particles,
            iterations=iterations,
        )

        all_results[customer_count] = results

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print_overall_summary(all_results)


if __name__ == "__main__":
    main()
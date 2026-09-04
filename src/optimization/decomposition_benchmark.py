import time

from src.algorithms.decomposed_warm_qpso import (
    DecomposedWarmQPSO,
)
from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark
from src.optimization.vrp_fitness import VRPFitnessFunction


def run_optimizer(name, optimizer):

    start = time.perf_counter()

    result = optimizer.optimize()

    execution_time = (
        time.perf_counter() - start
    )

    print(
        f"{name:<25}"
        f"fitness={result['fitness']:.4f}  "
        f"time={result['travel_time']:.4f}  "
        f"distance={result['distance']:.4f}  "
        f"runtime={execution_time:.2f}s"
    )

    return {
        "algorithm": name,
        "fitness": result["fitness"],
        "travel_time": result["travel_time"],
        "distance": result["distance"],
        "congestion": result["congestion"],
        "execution_time": execution_time,
        "result": result,
    }


def main():

    customer_count = 100
    seed = 42

    particles = 20
    iterations = 50

    print("\n")
    print("=" * 100)
    print("DECOMPOSED WARM-QPSO EXPERIMENT")
    print("=" * 100)

    print(f"Customers : {customer_count}")
    print(f"Particles : {particles}")
    print(f"Iterations: {iterations}")
    print(f"Seed      : {seed}")
    print("=" * 100)

    benchmark = VRPBenchmark(
        customer_sizes=(customer_count,),
        particles=particles,
        iterations=iterations,
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

    # ---------------------------------------------------------
    # PSO
    # ---------------------------------------------------------

    print("\nRunning PSO...")

    pso = VRPPSO(
        problem=problem,
        particles=particles,
        iterations=iterations,
        seed=seed,
        fitness_function=fitness_function,
    )

    results.append(
        run_optimizer("PSO", pso)
    )

    # ---------------------------------------------------------
    # QPSO
    # ---------------------------------------------------------

    print("\nRunning QPSO...")

    qpso = VRPQPSO(
        problem=problem,
        particles=particles,
        iterations=iterations,
        beta=0.7,
        seed=seed,
        fitness_function=fitness_function,
    )

    results.append(
        run_optimizer("QPSO", qpso)
    )

    # ---------------------------------------------------------
    # WARM-QPSO
    # ---------------------------------------------------------

    print("\nRunning Warm-QPSO...")

    warm_qpso = WarmVRPQPSO(
        problem=problem,
        particles=particles,
        iterations=iterations,
        beta=0.7,
        seed=seed,
        fitness_function=fitness_function,
        warm_iterations=10,
        warm_particles=10,
    )

    results.append(
        run_optimizer(
            "Warm-QPSO",
            warm_qpso,
        )
    )

    # ---------------------------------------------------------
    # DECOMPOSED WARM-QPSO
    # ---------------------------------------------------------

    print("\nRunning Decomposed Warm-QPSO...")

    decomposed = DecomposedWarmQPSO(
        problem=problem,
        clusters=5,
        particles=particles,
        iterations=iterations,
        beta=0.7,
        seed=seed,
        fitness_function=fitness_function,
        warm_iterations=10,
        warm_particles=10,
    )

    decomposed_result = run_optimizer(
        "Decomposed Warm-QPSO",
        decomposed,
    )

    results.append(
        decomposed_result
    )

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("FINAL COMPARISON")
    print("=" * 110)

    print(
        f"{'Algorithm':<25}"
        f"{'Fitness':>14}"
        f"{'Travel Time':>16}"
        f"{'Distance':>14}"
        f"{'Congestion':>14}"
        f"{'Runtime':>14}"
    )

    print("-" * 110)

    for result in results:

        print(
            f"{result['algorithm']:<25}"
            f"{result['fitness']:>14.4f}"
            f"{result['travel_time']:>16.4f}"
            f"{result['distance']:>14.4f}"
            f"{result['congestion']:>14.4f}"
            f"{result['execution_time']:>13.2f}s"
        )

    best = min(
        results,
        key=lambda result: result["fitness"],
    )

    print("-" * 110)

    print(
        f"\nBEST ALGORITHM: "
        f"{best['algorithm']}"
    )

    print(
        f"BEST FITNESS: "
        f"{best['fitness']:.4f}"
    )

    # ---------------------------------------------------------
    # CLUSTER INFORMATION
    # ---------------------------------------------------------

    if "clusters" in decomposed_result["result"]:

        print("\n")
        print("=" * 90)
        print("DECOMPOSITION DETAILS")
        print("=" * 90)

        for cluster in (
            decomposed_result["result"]["clusters"]
        ):

            print(
                f"\nCluster "
                f"{cluster['cluster_id']}"
            )

            print(
                f"Customers: "
                f"{len(cluster['customers'])}"
            )

            print(
                f"Vehicles: "
                f"{cluster['vehicles']}"
            )

            print(
                f"Fitness: "
                f"{cluster['fitness']:.4f}"
            )

            print(
                f"Warm-start fitness: "
                f"{cluster['warm_start_fitness']}"
            )

    print("\n")


if __name__ == "__main__":
    main()
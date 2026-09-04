"""
Exact VRP Benchmark
===================

Compare Q-ROUTE metaheuristics against an exact solver
on small VRP instances.

The exact solver provides the reference optimum.

The main metric is the optimality gap:

    gap (%) =
        ((algorithm_fitness - optimal_fitness)
         / optimal_fitness) * 100
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json

from src.optimization.config import OptimizationConfig
from src.optimization.vrp_fitness import VRPFitnessFunction
from src.optimization.exact_vrp import ExactVRPSolver

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.algorithms.decomposed_warm_qpso import (
    DecomposedWarmQPSO,
)

from dashboard.optimizer_service import (
    get_graph,
    build_problem,
)


RESULTS_DIR = Path("results")


def calculate_optimality_gap(
    fitness,
    optimal_fitness,
):
    """
    Calculate percentage optimality gap.
    """

    if optimal_fitness == 0:
        return 0.0

    return (
        (
            fitness
            - optimal_fitness
        )
        / optimal_fitness
    ) * 100.0


def run_exact_benchmark(
    customer_sizes=(5, 6, 7),
    vehicles=2,
    particles=10,
    iterations=20,
    seed=42,
):
    """
    Run exact-vs-metaheuristic benchmarking.

    Small instances are used because the exact solver
    enumerates the complete solution space.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for customer_count in customer_sizes:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"EXACT VRP BENCHMARK"
        )

        print(
            f"Customers : {customer_count}"
        )

        print(
            f"Vehicles  : {vehicles}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Build common problem
        # ----------------------------------------------------

        graph = get_graph(
            "Normal"
        )

        problem = build_problem(
            graph=graph,
            number_of_customers=customer_count,
            number_of_vehicles=vehicles,
        )

        fitness_function = (
            VRPFitnessFunction()
        )

        # ----------------------------------------------------
        # Exact solver
        # ----------------------------------------------------

        print(
            "\nRunning Exact Solver..."
        )

        exact_solver = ExactVRPSolver(
            problem=problem,
            fitness_function=fitness_function,
            max_customers=9,
        )

        exact_result = (
            exact_solver.solve()
        )

        optimal_fitness = (
            exact_result["fitness"]
        )

        print(
            f"Optimal Fitness: "
            f"{optimal_fitness:.4f}"
        )

        print(
            f"Exact Runtime: "
            f"{exact_result['execution_time']:.4f}s"
        )

        print(
            f"Solutions Evaluated: "
            f"{exact_result['evaluated_solutions']}"
        )

        # ----------------------------------------------------
        # Algorithm configurations
        # ----------------------------------------------------

        algorithms = [
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

            (
                "Decomposed Warm-QPSO",
                DecomposedWarmQPSO(
                    problem=problem,
                    clusters=min(
                        2,
                        customer_count,
                        vehicles,
                    ),
                    particles=particles,
                    iterations=iterations,
                    beta=0.7,
                    seed=seed,
                    fitness_function=fitness_function,
                    warm_iterations=10,
                    warm_particles=10,
                    repair=True,
                    repair_rounds=2,
                    max_boundary_customers=20,
                    max_target_vehicles=5,
                    try_swaps=True,
                ),
            ),
        ]

        # ----------------------------------------------------
        # Metaheuristic experiments
        # ----------------------------------------------------

        for algorithm_name, optimizer in algorithms:

            print(
                f"\nRunning {algorithm_name}..."
            )

            result = optimizer.optimize()

            gap = calculate_optimality_gap(
                result["fitness"],
                optimal_fitness,
            )

            print(
                f"Fitness : "
                f"{result['fitness']:.4f}"
            )

            print(
                f"Gap     : "
                f"{gap:.2f}%"
            )

            print(
                f"Runtime : "
                f"{result.get('execution_time', 0):.4f}s"
            )

            rows.append(
                {
                    "customers": customer_count,
                    "vehicles": vehicles,
                    "algorithm": algorithm_name,
                    "fitness": result["fitness"],
                    "optimal_fitness": optimal_fitness,
                    "optimality_gap_percent": gap,
                    "travel_time": result[
                        "travel_time"
                    ],
                    "distance": result[
                        "distance"
                    ],
                    "congestion": result[
                        "congestion"
                    ],
                    "execution_time": result.get(
                        "execution_time",
                        0.0,
                    ),
                    "valid": result.get(
                        "valid",
                        False,
                    ),
                }
            )

        # ----------------------------------------------------
        # Add exact solver result
        # ----------------------------------------------------

        rows.append(
            {
                "customers": customer_count,
                "vehicles": vehicles,
                "algorithm": "Exact",
                "fitness": exact_result[
                    "fitness"
                ],
                "optimal_fitness": optimal_fitness,
                "optimality_gap_percent": 0.0,
                "travel_time": exact_result[
                    "travel_time"
                ],
                "distance": exact_result[
                    "distance"
                ],
                "congestion": exact_result[
                    "congestion"
                ],
                "execution_time": exact_result[
                    "execution_time"
                ],
                "valid": exact_result[
                    "valid"
                ],
            }
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        RESULTS_DIR
        / "exact_vrp_benchmark.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            rows,
            file,
            indent=2,
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        RESULTS_DIR
        / "exact_vrp_benchmark.csv"
    )

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EXACT VRP BENCHMARK COMPLETE"
    )

    print(
        f"Saved: {json_path}"
    )

    print(
        f"Saved: {csv_path}"
    )

    print(
        "=" * 60
    )

    return rows


if __name__ == "__main__":

    config = OptimizationConfig(
        particles=10,
        iterations=20,
        seed=42,
    )

    run_exact_benchmark(
        customer_sizes=(5, 6, 7),
        vehicles=2,
        particles=config.particles,
        iterations=config.iterations,
        seed=config.seed,
    )
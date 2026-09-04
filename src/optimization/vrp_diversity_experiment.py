"""
PSO vs QPSO diversity experiment for VRP.

Measures how many distinct customer permutations are explored
during optimization.

The experiment uses the same:
    - VRP problem
    - fitness function
    - number of particles
    - iterations
    - random seed

for PSO and QPSO.
"""

import csv
import os

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark


def permutation_from_position(algorithm, position):
    """
    Convert a continuous position into a customer permutation.
    """

    return tuple(
        algorithm._decode_customer_order(position)
    )


def run_algorithm(
    algorithm_name,
    problem,
    fitness_function,
    seed,
):
    """
    Run an algorithm and record diversity at every iteration.
    """

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

    # ------------------------------------------------------------
    # Initialize swarm.
    # ------------------------------------------------------------

    swarm = algorithm._initialize_particles()

    global_best = algorithm._find_global_best(
        swarm
    )

    if global_best is None:
        raise RuntimeError(
            "No feasible initial particle found."
        )

    if algorithm_name == "QPSO":

        global_best_position = list(
            global_best["best_position"]
        )

    else:

        global_best_position = list(
            global_best["best_position"]
        )

    records = []

    # ------------------------------------------------------------
    # Record initial state.
    # ------------------------------------------------------------

    records.append(
        collect_diversity_record(
            algorithm_name,
            problem.number_of_customers,
            0,
            swarm,
            algorithm,
        )
    )

    # ------------------------------------------------------------
    # Optimization loop.
    # ------------------------------------------------------------

    for iteration in range(1, 51):

        if algorithm_name == "PSO":

            # Classical PSO update.
            for particle in swarm:

                current_position = (
                    particle["position"]
                )

                velocity = (
                    particle["velocity"]
                )

                new_velocity = []

                for i in range(
                    algorithm.dimension
                ):

                    r1 = algorithm.random.random()
                    r2 = algorithm.random.random()

                    value = (
                        algorithm.inertia
                        * velocity[i]
                        + algorithm.cognitive
                        * r1
                        * (
                            particle[
                                "best_position"
                            ][i]
                            - current_position[i]
                        )
                        + algorithm.social
                        * r2
                        * (
                            global_best_position[i]
                            - current_position[i]
                        )
                    )

                    value = max(
                        -1.0,
                        min(1.0, value),
                    )

                    new_velocity.append(
                        value
                    )

                new_position = [
                    max(
                        0.0,
                        min(
                            1.0,
                            current_position[i]
                            + new_velocity[i],
                        ),
                    )
                    for i in range(
                        algorithm.dimension
                    )
                ]

                evaluation = (
                    algorithm.evaluate_position(
                        new_position
                    )
                )

                particle["velocity"] = (
                    new_velocity
                )

                particle["position"] = (
                    new_position
                )

                particle["fitness"] = (
                    evaluation["fitness"]
                )

                if (
                    particle["fitness"]
                    < particle["best_fitness"]
                ):

                    particle["best_position"] = (
                        list(new_position)
                    )

                    particle["best_fitness"] = (
                        particle["fitness"]
                    )

                if (
                    particle["best_fitness"]
                    < global_best["best_fitness"]
                ):

                    global_best = particle

                    global_best_position = (
                        list(
                            particle[
                                "best_position"
                            ]
                        )
                    )

        else:

            mean_best = (
                algorithm._calculate_mean_best(
                    swarm,
                    algorithm.dimension,
                )
            )

            for particle in swarm:

                new_position = (
                    algorithm._update_position(
                        position=particle[
                            "position"
                        ],
                        best_position=particle[
                            "best_position"
                        ],
                        mean_best=mean_best,
                        global_best_position=(
                            global_best_position
                        ),
                    )
                )

                evaluation = (
                    algorithm.evaluate_position(
                        new_position
                    )
                )

                particle["position"] = (
                    new_position
                )

                particle["fitness"] = (
                    evaluation["fitness"]
                )

                if (
                    particle["fitness"]
                    < particle["best_fitness"]
                ):

                    particle["best_position"] = (
                        list(new_position)
                    )

                    particle["best_fitness"] = (
                        particle["fitness"]
                    )

                if (
                    particle["best_fitness"]
                    < global_best["best_fitness"]
                ):

                    global_best = particle

                    global_best_position = (
                        list(
                            particle[
                                "best_position"
                            ]
                        )
                    )

        records.append(
            collect_diversity_record(
                algorithm_name,
                problem.number_of_customers,
                iteration,
                swarm,
                algorithm,
            )
        )

    return records


def collect_diversity_record(
    algorithm_name,
    customer_count,
    iteration,
    swarm,
    algorithm,
):
    """
    Calculate diversity metrics for one iteration.
    """

    permutations = []

    fitness_values = []

    for particle in swarm:

        position = particle["position"]

        permutation = (
            permutation_from_position(
                algorithm,
                position,
            )
        )

        permutations.append(
            permutation
        )

        fitness = particle["fitness"]

        if fitness != float("inf"):
            fitness_values.append(
                fitness
            )

    unique_permutations = len(
        set(permutations)
    )

    if fitness_values:

        average_fitness = (
            sum(fitness_values)
            / len(fitness_values)
        )

        best_fitness = min(
            fitness_values
        )

    else:

        average_fitness = float("inf")
        best_fitness = float("inf")

    return {
        "algorithm": algorithm_name,
        "customers": customer_count,
        "iteration": iteration,
        "unique_permutations": (
            unique_permutations
        ),
        "average_fitness": (
            average_fitness
        ),
        "best_fitness": (
            best_fitness
        ),
    }


def main():

    customer_sizes = (
        10,
        20,
        50,
    )

    seed = 42

    benchmark = VRPBenchmark(
        customer_sizes=(10,),
        particles=20,
        iterations=50,
        seed=42,
    )

    all_records = []

    print("\nVRP diversity experiment")
    print("========================")

    for customer_count in customer_sizes:

        print(
            f"\n--- {customer_count} customers ---"
        )

        problem = benchmark._build_problem(
            customer_count
        )

        for algorithm_name in (
            "PSO",
            "QPSO",
        ):

            records = run_algorithm(
                algorithm_name,
                problem,
                benchmark.fitness_function,
                seed,
            )

            all_records.extend(
                records
            )

            initial = records[0]
            final = records[-1]

            print(
                f"{algorithm_name:5s} | "
                f"initial unique="
                f"{initial['unique_permutations']} | "
                f"final unique="
                f"{final['unique_permutations']} | "
                f"final fitness="
                f"{final['best_fitness']:.4f}"
            )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    output_path = (
        "results/vrp_diversity.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "algorithm",
            "customers",
            "iteration",
            "unique_permutations",
            "average_fitness",
            "best_fitness",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            all_records
        )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
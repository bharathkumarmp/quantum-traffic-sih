import math
import random
import statistics
from src.optimization.vrp_benchmark import VRPBenchmark
from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO


def permutation_distance(a, b):
    """Number of positions whose customers changed."""
    if a is None or b is None:
        return 0

    return sum(x != y for x, y in zip(a, b))


def run_qpso(problem, particles=20, iterations=50, beta=0.7, seed=42):
    optimizer = VRPQPSO(
        problem,
        particles=particles,
        iterations=iterations,
        beta=beta,
        seed=seed,
    )

    # Reproduce the optimizer's initialization
    particle_list = []

    for _ in range(particles):
        position = [
            optimizer.random.random()
            for _ in range(problem.number_of_customers)
        ]

        fitness_result = optimizer.evaluate_position(position)

        particle_list.append({
            "position": position,
            "best_position": position[:],
            "best_fitness": fitness_result["fitness"],
        })

    global_best = min(
        particle_list,
        key=lambda p: p["best_fitness"]
    )

    previous_permutations = []

    records = []

    for iteration in range(iterations + 1):

        current_permutations = []

        for particle in particle_list:
            permutation = tuple(
                optimizer._decode_customer_order(
                    particle["position"]
                )
            )
            current_permutations.append(permutation)

        unique_count = len(set(current_permutations))

        if iteration == 0:
            changes = 0
            avg_position_movement = 0.0
        else:
            changes = sum(
                permutation_distance(
                    previous_permutations[i],
                    current_permutations[i]
                )
                for i in range(particles)
            ) / particles

            avg_position_movement = statistics.mean(
                particle["_movement"]
                for particle in particle_list
            )

        fitness_values = [
            particle["best_fitness"]
            for particle in particle_list
        ]

        records.append({
            "iteration": iteration,
            "unique_permutations": unique_count,
            "average_fitness": statistics.mean(fitness_values),
            "best_fitness": min(fitness_values),
            "average_permutation_changes": changes,
            "average_position_movement": avg_position_movement,
        })

        if iteration == iterations:
            break

        previous_permutations = current_permutations

        best_positions = [
            particle["best_position"]
            for particle in particle_list
        ]

        mean_best = [
            statistics.mean(
                position[j]
                for position in best_positions
            )
            for j in range(problem.number_of_customers)
        ]

        global_best_position = global_best["best_position"]

        for particle in particle_list:

            old_position = particle["position"][:]

            new_position = optimizer._update_position(
                particle["position"],
                particle["best_position"],
                mean_best,
                global_best_position,
            )

            movement = statistics.mean(
                abs(new_position[j] - old_position[j])
                for j in range(len(old_position))
            )

            particle["_movement"] = movement
            particle["position"] = new_position

            result = optimizer.evaluate_position(new_position)

            if result["fitness"] < particle["best_fitness"]:
                particle["best_fitness"] = result["fitness"]
                particle["best_position"] = new_position[:]

                if (
                    particle["best_fitness"]
                    < global_best["best_fitness"]
                ):
                    global_best = particle


    return records


def main():

    print()
    print("VRP QPSO dynamics experiment")
    print("============================")

    benchmark = VRPBenchmark(
        customer_sizes=(10, 20, 50),
        particles=20,
        iterations=50,
        seed=42,
    )

    for size in (10, 20, 50):

        print()
        print(f"--- {size} customers ---")

        problem = benchmark._build_problem(size)

        records = run_qpso(
            problem,
            particles=20,
            iterations=50,
            beta=0.7,
            seed=42,
        )

        print(
            "Iteration | Unique | Avg Fitness | "
            "Best Fitness | Perm Changes | Position Move"
        )
        print("-" * 75)

        selected = {
            0,
            1,
            5,
            10,
            20,
            30,
            40,
            50,
        }

        for record in records:

            if record["iteration"] in selected:

                print(
                    f"{record['iteration']:9d} | "
                    f"{record['unique_permutations']:6d} | "
                    f"{record['average_fitness']:11.4f} | "
                    f"{record['best_fitness']:12.4f} | "
                    f"{record['average_permutation_changes']:13.2f} | "
                    f"{record['average_position_movement']:14.6f}"
                )

        initial = records[0]
        final = records[-1]

        print()
        print("Summary")
        print(
            f"Initial unique permutations : "
            f"{initial['unique_permutations']}"
        )
        print(
            f"Final unique permutations   : "
            f"{final['unique_permutations']}"
        )
        print(
            f"Final best fitness          : "
            f"{final['best_fitness']:.4f}"
        )
        print(
            f"Final avg fitness           : "
            f"{final['average_fitness']:.4f}"
        )
        print(
            f"Final permutation changes   : "
            f"{final['average_permutation_changes']:.2f}"
        )
        print(
            f"Final position movement     : "
            f"{final['average_position_movement']:.6f}"
        )


if __name__ == "__main__":
    main()
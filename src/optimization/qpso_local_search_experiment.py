import random
import time

from src.optimization.vrp_benchmark import VRPBenchmark
from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO


def swap_move(order, i, j):
    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order


def insertion_move(order, i, j):
    new_order = order[:]
    customer = new_order.pop(i)
    new_order.insert(j, customer)
    return new_order


def reversal_move(order, i, j):
    new_order = order[:]
    new_order[i:j + 1] = reversed(new_order[i:j + 1])
    return new_order


def permutation_to_position(order, reference_order):
    """
    Convert a customer permutation back into random-key positions.

    The relative ordering of the keys reproduces the permutation.
    """
    n = len(order)
    position = [0.0] * n

    rank = {
        customer: index
        for index, customer in enumerate(order)
    }

    for index, customer in enumerate(reference_order):
        position[index] = (rank[customer] + 1) / (n + 1)

    return position


def local_search(optimizer, position):
    """
    Simple first-improvement discrete local search.

    Tests:
      1. swap
      2. insertion
      3. reversal

    Returns an improved random-key position if one is found.
    """

    current_position = position[:]

    current_order = optimizer._decode_customer_order(
        current_position
    )

    current_result = optimizer.evaluate_position(
        current_position
    )

    current_fitness = current_result["fitness"]

    n = len(current_order)

    # Limit neighborhood size for larger VRPs.
    max_moves = min(200, n * 10)

    moves_checked = 0

    # Deterministic RNG from optimizer seed/state.
    move_indices = list(range(n))
    optimizer.random.shuffle(move_indices)

    # -------------------------
    # Swap neighborhood
    # -------------------------
    for a in range(n):
        for b in range(a + 1, n):

            candidate_order = swap_move(
                current_order,
                a,
                b
            )

            candidate_position = permutation_to_position(
                candidate_order,
                current_order
            )

            result = optimizer.evaluate_position(
                candidate_position
            )

            moves_checked += 1

            if result["valid"] and result["fitness"] < current_fitness:
                return candidate_position

            if moves_checked >= max_moves:
                break

        if moves_checked >= max_moves:
            break

    # -------------------------
    # Insertion neighborhood
    # -------------------------
    moves_checked = 0

    for a in range(n):
        for b in range(n):

            if a == b:
                continue

            candidate_order = insertion_move(
                current_order,
                a,
                b
            )

            candidate_position = permutation_to_position(
                candidate_order,
                current_order
            )

            result = optimizer.evaluate_position(
                candidate_position
            )

            moves_checked += 1

            if result["valid"] and result["fitness"] < current_fitness:
                return candidate_position

            if moves_checked >= max_moves:
                break

        if moves_checked >= max_moves:
            break

    # -------------------------
    # Reversal neighborhood
    # -------------------------
    moves_checked = 0

    for a in range(n):
        for b in range(a + 1, n):

            candidate_order = reversal_move(
                current_order,
                a,
                b
            )

            candidate_position = permutation_to_position(
                candidate_order,
                current_order
            )

            result = optimizer.evaluate_position(
                candidate_position
            )

            moves_checked += 1

            if result["valid"] and result["fitness"] < current_fitness:
                return candidate_position

            if moves_checked >= max_moves:
                break

        if moves_checked >= max_moves:
            break

    return current_position


def run_qpso(problem, particles=20, iterations=50, beta=0.7, seed=42,
             use_local_search=False):

    optimizer = VRPQPSO(
        problem,
        particles=particles,
        iterations=iterations,
        beta=beta,
        seed=seed,
    )

    particle_list = []

    # -------------------------
    # Initialization
    # -------------------------
    for _ in range(particles):

        position = [
            optimizer.random.random()
            for _ in range(problem.number_of_customers)
        ]

        result = optimizer.evaluate_position(position)

        particle_list.append({
            "position": position,
            "best_position": position[:],
            "best_fitness": result["fitness"],
        })

    global_best = min(
        particle_list,
        key=lambda p: p["best_fitness"]
    )

    # -------------------------
    # Optimization
    # -------------------------
    for iteration in range(iterations):

        best_positions = [
            particle["best_position"]
            for particle in particle_list
        ]

        mean_best = [
            sum(
                position[j]
                for position in best_positions
            ) / particles
            for j in range(problem.number_of_customers)
        ]

        global_best_position = global_best["best_position"][:]

        for particle in particle_list:

            new_position = optimizer._update_position(
                particle["position"],
                particle["best_position"],
                mean_best,
                global_best_position,
            )

            # Optional local search
            if use_local_search:
                new_position = local_search(
                    optimizer,
                    new_position
                )

            result = optimizer.evaluate_position(
                new_position
            )

            particle["position"] = new_position

            if result["fitness"] < particle["best_fitness"]:

                particle["best_fitness"] = result["fitness"]
                particle["best_position"] = new_position[:]

                if (
                    particle["best_fitness"]
                    < global_best["best_fitness"]
                ):
                    global_best = particle

    final_result = optimizer.evaluate_position(
        global_best["best_position"]
    )

    return final_result


def run_pso(problem, particles=20, iterations=50, seed=42):

    optimizer = VRPPSO(
        problem,
        particles=particles,
        iterations=iterations,
        seed=seed,
    )

    start = time.perf_counter()

    result = optimizer.optimize()

    elapsed = time.perf_counter() - start

    return result, elapsed


def main():

    print()
    print("QPSO + Local Search Ablation")
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

        # -------------------------
        # Classical PSO
        # -------------------------
        pso = VRPPSO(
            problem,
            particles=20,
            iterations=50,
            seed=42,
        )

        pso_start = time.perf_counter()
        pso_result = pso.optimize()
        pso_time = time.perf_counter() - pso_start

        # -------------------------
        # QPSO
        # -------------------------
        qpso_start = time.perf_counter()

        qpso_result = run_qpso(
            problem,
            particles=20,
            iterations=50,
            beta=0.7,
            seed=42,
            use_local_search=False,
        )

        qpso_time = time.perf_counter() - qpso_start

        # -------------------------
        # QPSO + Local Search
        # -------------------------
        hybrid_start = time.perf_counter()

        hybrid_result = run_qpso(
            problem,
            particles=20,
            iterations=50,
            beta=0.7,
            seed=42,
            use_local_search=True,
        )

        hybrid_time = time.perf_counter() - hybrid_start

        print()
        print(
            f"{'Algorithm':<22}"
            f"{'Fitness':>12}"
            f"{'Time':>12}"
            f"{'Distance':>12}"
            f"{'Execution':>12}"
        )

        print("-" * 70)

        print(
            f"{'PSO':<22}"
            f"{pso_result['fitness']:>12.4f}"
            f"{pso_result['travel_time']:>12.4f}"
            f"{pso_result['distance']:>12.4f}"
            f"{pso_time:>12.3f}"
        )

        print(
            f"{'QPSO':<22}"
            f"{qpso_result['fitness']:>12.4f}"
            f"{qpso_result['travel_time']:>12.4f}"
            f"{qpso_result['distance']:>12.4f}"
            f"{qpso_time:>12.3f}"
        )

        print(
            f"{'QPSO + Local Search':<22}"
            f"{hybrid_result['fitness']:>12.4f}"
            f"{hybrid_result['travel_time']:>12.4f}"
            f"{hybrid_result['distance']:>12.4f}"
            f"{hybrid_time:>12.3f}"
        )


if __name__ == "__main__":
    main()
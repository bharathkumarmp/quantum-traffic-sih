import os
import sys
import csv
import time

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.data.graph_generator import LargeGraphGenerator
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO
from src.optimization.fitness import FitnessFunction


# =========================================================
# CONFIGURATION
# =========================================================

NUM_NODES = 1000

EDGES_PER_NODE = 3

PARTICLES = 20

ITERATIONS = 50

SEED = 42

START = "N0"

DESTINATION = f"N{NUM_NODES - 1}"

RESULT_DIR = "results"

CSV_FILE = os.path.join(
    RESULT_DIR,
    "convergence_analysis.csv"
)


# =========================================================
# GENERATE NETWORK
# =========================================================

print("=" * 70)
print("CONVERGENCE ANALYSIS")
print("=" * 70)

print(
    f"\nNetwork nodes : {NUM_NODES:,}"
)

print(
    f"Edges per node: {EDGES_PER_NODE}"
)

print(
    f"Particles     : {PARTICLES}"
)

print(
    f"Iterations    : {ITERATIONS}"
)

print(
    f"Seed          : {SEED}"
)


print("\nGenerating network...")

generation_start = time.perf_counter()

generator = LargeGraphGenerator(
    num_nodes=NUM_NODES,
    edges_per_node=EDGES_PER_NODE,
    seed=SEED
)

graph = generator.generate()

generation_time = (
    time.perf_counter()
    - generation_start
)

print(
    f"Generation time: "
    f"{generation_time:.6f}s"
)

print(
    f"Nodes          : "
    f"{graph.number_of_nodes():,}"
)

print(
    f"Edges          : "
    f"{len(graph.get_edges()):,}"
)


# =========================================================
# COMMON FITNESS
# =========================================================

fitness_function = FitnessFunction()


# =========================================================
# RUN PSO
# =========================================================

print("\n" + "=" * 70)
print("RUNNING PSO")
print("=" * 70)

pso = PSO(
    graph=graph,
    particles=PARTICLES,
    iterations=ITERATIONS,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    seed=SEED,
    fitness_function=fitness_function,
)

pso_start = time.perf_counter()

pso_result = pso.optimize(
    START,
    DESTINATION
)

pso_runtime = (
    time.perf_counter()
    - pso_start
)

print(
    f"Runtime      : "
    f"{pso_runtime:.6f}s"
)

print(
    f"Final fitness: "
    f"{pso_result['fitness']:.6f}"
)

print(
    f"Iterations   : "
    f"{pso_result['iterations']}"
)


# =========================================================
# RUN QPSO
# =========================================================

print("\n" + "=" * 70)
print("RUNNING QPSO")
print("=" * 70)

qpso = QPSO(
    graph=graph,
    particles=PARTICLES,
    iterations=ITERATIONS,
    beta=0.7,
    seed=SEED,
    fitness_function=fitness_function,
)

qpso_start = time.perf_counter()

qpso_result = qpso.optimize(
    START,
    DESTINATION
)

qpso_runtime = (
    time.perf_counter()
    - qpso_start
)

print(
    f"Runtime      : "
    f"{qpso_runtime:.6f}s"
)

print(
    f"Final fitness: "
    f"{qpso_result['fitness']:.6f}"
)

print(
    f"Iterations   : "
    f"{qpso_result['iterations']}"
)


# =========================================================
# EXTRACT HISTORY
# =========================================================

pso_history = pso_result["history"]

qpso_history = qpso_result["history"]


# =========================================================
# PRINT CONVERGENCE TABLE
# =========================================================

print("\n" + "=" * 70)
print("CONVERGENCE HISTORY")
print("=" * 70)

print(
    f"{'Iteration':>12}"
    f"{'PSO Fitness':>20}"
    f"{'QPSO Fitness':>20}"
)

print("-" * 70)

max_iterations = max(
    len(pso_history),
    len(qpso_history)
)

for i in range(max_iterations):

    pso_value = (
        pso_history[i]
        if i < len(pso_history)
        else pso_history[-1]
    )

    qpso_value = (
        qpso_history[i]
        if i < len(qpso_history)
        else qpso_history[-1]
    )

    print(
        f"{i + 1:>12}"
        f"{pso_value:>20.6f}"
        f"{qpso_value:>20.6f}"
    )


# =========================================================
# IMPROVEMENT CALCULATION
# =========================================================

pso_initial = pso_history[0]

pso_final = pso_history[-1]

qpso_initial = qpso_history[0]

qpso_final = qpso_history[-1]


if pso_initial != 0:

    pso_improvement = (
        (pso_initial - pso_final)
        / abs(pso_initial)
    ) * 100

else:

    pso_improvement = 0.0


if qpso_initial != 0:

    qpso_improvement = (
        (qpso_initial - qpso_final)
        / abs(qpso_initial)
    ) * 100

else:

    qpso_improvement = 0.0


# =========================================================
# CONVERGENCE ITERATION
# =========================================================

def find_convergence_iteration(
    history,
    tolerance=1e-6,
    window=5
):

    if len(history) <= window:
        return len(history)

    for i in range(window, len(history)):

        previous = history[
            i - window
        ]

        current = history[i]

        improvement = abs(
            previous - current
        )

        if improvement <= tolerance:

            return i + 1

    return len(history)


pso_convergence = (
    find_convergence_iteration(
        pso_history
    )
)

qpso_convergence = (
    find_convergence_iteration(
        qpso_history
    )
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("CONVERGENCE SUMMARY")
print("=" * 70)

print(
    f"PSO initial fitness       : "
    f"{pso_initial:.6f}"
)

print(
    f"PSO final fitness         : "
    f"{pso_final:.6f}"
)

print(
    f"PSO improvement           : "
    f"{pso_improvement:.2f}%"
)

print(
    f"PSO convergence iteration : "
    f"{pso_convergence}"
)

print(
    f"PSO runtime               : "
    f"{pso_runtime:.6f}s"
)

print()

print(
    f"QPSO initial fitness       : "
    f"{qpso_initial:.6f}"
)

print(
    f"QPSO final fitness         : "
    f"{qpso_final:.6f}"
)

print(
    f"QPSO improvement           : "
    f"{qpso_improvement:.2f}%"
)

print(
    f"QPSO convergence iteration : "
    f"{qpso_convergence}"
)

print(
    f"QPSO runtime               : "
    f"{qpso_runtime:.6f}s"
)


# =========================================================
# SAVE CSV
# =========================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

with open(
    CSV_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "iteration",
        "pso_fitness",
        "qpso_fitness",
    ])

    for i in range(max_iterations):

        pso_value = (
            pso_history[i]
            if i < len(pso_history)
            else pso_history[-1]
        )

        qpso_value = (
            qpso_history[i]
            if i < len(qpso_history)
            else qpso_history[-1]
        )

        writer.writerow([
            i + 1,
            pso_value,
            qpso_value,
        ])


print("\n" + "=" * 70)

print(
    "Convergence data saved to:"
)

print(
    CSV_FILE
)

print("=" * 70)
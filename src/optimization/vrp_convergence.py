import csv
import os

from src.optimization.vrp_benchmark import VRPBenchmark


def main():
    benchmark = VRPBenchmark(
        customer_sizes=(10, 20, 50, 100),
        particles=20,
        iterations=50,
        seed=42,
    )

    results = benchmark.run()

    os.makedirs("results", exist_ok=True)

    rows = []

    for result in results:
        history = result["history"]

        for iteration, fitness in enumerate(history):
            rows.append(
                {
                    "algorithm": result["algorithm"],
                    "customers": result["customers"],
                    "iteration": iteration,
                    "fitness": fitness,
                }
            )

    output_path = "results/vrp_convergence.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "customers",
                "iteration",
                "fitness",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nConvergence experiment completed.")
    print(f"Saved: {output_path}")

    print("\nFinal convergence values:")

    for result in results:
        history = result["history"]

        print(
            f"{result['algorithm']:5s} | "
            f"{result['customers']:3d} customers | "
            f"initial={history[0]:.4f} | "
            f"final={history[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
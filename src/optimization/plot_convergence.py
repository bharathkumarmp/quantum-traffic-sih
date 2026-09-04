import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_convergence():
    project_root = Path(__file__).resolve().parents[2]

    csv_path = (
        project_root
        / "results"
        / "convergence_benchmark.csv"
    )

    output_path = (
        project_root
        / "results"
        / "convergence_plot.png"
    )

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))

    for algorithm in df["algorithm"].unique():
        data = df[df["algorithm"] == algorithm]

        plt.plot(
            data["iteration"],
            data["fitness"],
            linewidth=2,
            label=algorithm,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Global Fitness")

    plt.title(
        "Q-ROUTE Optimization Convergence"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    plot_convergence()
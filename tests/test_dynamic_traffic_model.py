"""
Test Dynamic Traffic Model
"""

from src.data.graph_generator import LargeGraphGenerator
from src.traffic.traffic_model import (
    DynamicTrafficModel,
)


def main():

    generator = LargeGraphGenerator(
        num_nodes=10,
        edges_per_node=3,
        seed=42,
    )

    graph = generator.generate()

    traffic = DynamicTrafficModel(
        seed=42,
    )

    print()
    print("=" * 60)
    print("DYNAMIC TRAFFIC MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    traffic.normal(graph)

    print()
    print("Normal traffic:")

    for edge in graph.get_edges()[:5]:

        print(
            f"{edge.source.id} -> "
            f"{edge.target.id} | "
            f"congestion="
            f"{edge.congestion:.3f} | "
            f"time="
            f"{edge.travel_time:.3f}"
        )

    # --------------------------------------------------------
    # Gradual congestion
    # --------------------------------------------------------

    traffic.gradual_congestion(
        graph,
        level=0.5,
    )

    print()
    print("Gradual congestion (50%):")

    for edge in graph.get_edges()[:5]:

        print(
            f"{edge.source.id} -> "
            f"{edge.target.id} | "
            f"congestion="
            f"{edge.congestion:.3f} | "
            f"time="
            f"{edge.travel_time:.3f}"
        )

    # --------------------------------------------------------
    # Random traffic
    # --------------------------------------------------------

    traffic.random_traffic(
        graph,
        variation=0.15,
    )

    print()
    print("Random traffic:")

    for edge in graph.get_edges()[:5]:

        print(
            f"{edge.source.id} -> "
            f"{edge.target.id} | "
            f"congestion="
            f"{edge.congestion:.3f} | "
            f"time="
            f"{edge.travel_time:.3f}"
        )

    print()
    print(
        "Dynamic traffic model test completed."
    )


if __name__ == "__main__":
    main()
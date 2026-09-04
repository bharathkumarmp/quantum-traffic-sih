import random

from src.domain.node import Node
from src.domain.edge import Edge
from src.domain.graph import Graph


class LargeGraphGenerator:
    """
    Generates reproducible synthetic transportation networks.

    Supports scalable graph generation for:

    - shortest-path experiments
    - PSO/QPSO benchmarking
    - dynamic traffic experiments
    - scalability testing
    """

    def __init__(
        self,
        num_nodes,
        edges_per_node=3,
        seed=42,
    ):
        if num_nodes < 2:
            raise ValueError(
                "num_nodes must be at least 2."
            )

        if edges_per_node < 1:
            raise ValueError(
                "edges_per_node must be at least 1."
            )

        self.num_nodes = int(num_nodes)
        self.edges_per_node = int(edges_per_node)
        self.seed = seed

        self.random = random.Random(seed)

    def generate(self):
        """
        Generate and return a transportation graph.
        """

        graph = Graph()

        nodes = [
            Node(f"N{i}")
            for i in range(self.num_nodes)
        ]

        # -------------------------------------------------
        # CONNECTIVITY BACKBONE
        # -------------------------------------------------
        #
        # Create:
        #
        # N0 -> N1 -> N2 -> N3 -> ...
        #
        # This guarantees that the generated graph
        # has a connected forward path.
        #

        for i in range(self.num_nodes - 1):

            distance = self.random.uniform(
                1.0,
                10.0,
            )

            congestion = self.random.uniform(
                0.0,
                0.3,
            )

            travel_time = (
                distance
                * self.random.uniform(
                    0.8,
                    1.5,
                )
                * (1.0 + congestion)
            )

            graph.add_edge(
                Edge(
                    nodes[i],
                    nodes[i + 1],
                    distance=distance,
                    travel_time=travel_time,
                    congestion=congestion,
                )
            )

        # -------------------------------------------------
        # RANDOM ROAD CONNECTIONS
        # -------------------------------------------------

        for source_index in range(
            self.num_nodes
        ):

            candidates = list(
                range(self.num_nodes)
            )

            candidates.remove(source_index)

            max_edges = min(
                self.edges_per_node,
                len(candidates),
            )

            targets = self.random.sample(
                candidates,
                max_edges,
            )

            for target_index in targets:

                distance = self.random.uniform(
                    1.0,
                    10.0,
                )

                congestion = self.random.uniform(
                    0.0,
                    0.3,
                )

                travel_time = (
                    distance
                    * self.random.uniform(
                        0.8,
                        1.5,
                    )
                    * (1.0 + congestion)
                )

                graph.add_edge(
                    Edge(
                        nodes[source_index],
                        nodes[target_index],
                        distance=distance,
                        travel_time=travel_time,
                        congestion=congestion,
                    )
                )

        return graph
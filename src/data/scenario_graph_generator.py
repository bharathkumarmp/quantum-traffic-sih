from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node


class ScenarioGraphGenerator:
    """
    Creates a small controlled transportation network with
    two competing routes.

    Network:

                N1 -------- N2
               /              \
        N0 ---                  --- N5
               \              /
                N3 -------- N4

    Upper route:
        N0 -> N1 -> N2 -> N5

    Lower route:
        N0 -> N3 -> N4 -> N5
    """

    def __init__(self):
        self.nodes = {
            node_id: Node(node_id)
            for node_id in (
                "N0",
                "N1",
                "N2",
                "N3",
                "N4",
                "N5",
            )
        }

    def _edge(
        self,
        source,
        target,
        distance,
        travel_time,
        congestion=0.1,
    ):
        return Edge(
            self.nodes[source],
            self.nodes[target],
            distance=distance,
            travel_time=travel_time,
            congestion=congestion,
        )

    def generate(self):
        """Generate the normal network."""

        graph = Graph()

        # Upper route
        graph.add_edge(
            self._edge(
                "N0",
                "N1",
                5.0,
                5.0,
                0.10,
            )
        )

        graph.add_edge(
            self._edge(
                "N1",
                "N2",
                5.0,
                5.0,
                0.10,
            )
        )

        graph.add_edge(
            self._edge(
                "N2",
                "N5",
                5.0,
                5.0,
                0.10,
            )
        )

        # Lower route
        graph.add_edge(
            self._edge(
                "N0",
                "N3",
                6.0,
                6.0,
                0.10,
            )
        )

        graph.add_edge(
            self._edge(
                "N3",
                "N4",
                6.0,
                6.0,
                0.10,
            )
        )

        graph.add_edge(
            self._edge(
                "N4",
                "N5",
                6.0,
                6.0,
                0.10,
            )
        )

        return graph

    def upper_route_incident(self):
        """
        Make the upper route expensive.

        N1 -> N2 becomes heavily congested.
        """

        graph = self.generate()

        for edge in graph.get_edges():
            if (
                edge.source.id == "N1"
                and edge.target.id == "N2"
            ):
                edge.travel_time = 30.0
                edge.congestion = 0.95

        return graph

    def upper_route_blockage(self):
        """
        Block the central edge of the upper route.
        """

        graph = self.generate()

        for edge in graph.get_edges():
            if (
                edge.source.id == "N1"
                and edge.target.id == "N2"
            ):
                edge.blocked = True

        return graph

    def lower_route_incident(self):
        """
        Make the lower route expensive.
        """

        graph = self.generate()

        for edge in graph.get_edges():
            if (
                edge.source.id == "N3"
                and edge.target.id == "N4"
            ):
                edge.travel_time = 30.0
                edge.congestion = 0.95

        return graph
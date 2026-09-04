import copy
import random

from src.traffic.scenario import (
    Scenario,
    TrafficScenario,
)


class TrafficModel:
    """
    Static traffic scenario manager.

    The original graph is preserved when using
    apply_scenario().
    """

    def __init__(self, graph):
        if graph is None:
            raise ValueError(
                "graph cannot be None"
            )

        self.graph = graph

    # =========================================================
    # APPLY SCENARIO
    # =========================================================

    def apply_scenario(self, scenario):
        """
        Apply a scenario to a copy of self.graph.

        The original graph is not modified.
        """

        if scenario is None:
            raise ValueError(
                "scenario cannot be None"
            )

        if not hasattr(scenario, "apply"):
            raise TypeError(
                "scenario must have an apply(graph) method."
            )

        updated_graph = copy.deepcopy(
            self.graph
        )

        scenario.apply(updated_graph)

        return updated_graph

    # =========================================================
    # NORMAL
    # =========================================================

    def normal_scenario(self):
        return Scenario(
            Scenario.NORMAL
        )

    def normal_traffic_scenario(self):
        return self.normal_scenario()

    def normal_traffic(self):
        return self.normal_scenario()

    # =========================================================
    # PEAK
    # =========================================================

    def peak_scenario(self):
        return Scenario(
            Scenario.PEAK
        )

    def peak_traffic_scenario(self):
        return self.peak_scenario()

    def heavy_traffic_scenario(self):
        return Scenario(
            Scenario.PEAK
        )

    # =========================================================
    # CONGESTED
    # =========================================================

    def congested_scenario(self):
        return Scenario(
            Scenario.CONGESTED
        )

    def congested_traffic_scenario(self):
        return self.congested_scenario()

    # =========================================================
    # INCIDENT
    # =========================================================

    def incident_scenario(
        self,
        source,
        target,
        travel_time_multiplier=3.0,
        congestion=0.95,
    ):
        return Scenario(
            Scenario.INCIDENT,
            source=source,
            target=target,
            travel_time_multiplier=(
                travel_time_multiplier
            ),
            congestion=congestion,
        )

    # =========================================================
    # ROAD BLOCKAGE
    # =========================================================

    def road_blockage_scenario(
        self,
        source,
        target,
    ):
        return Scenario(
            Scenario.ROAD_BLOCKAGE,
            source=source,
            target=target,
            blocked=True,
        )

    # =========================================================
    # DIRECT INCIDENT
    # =========================================================

    def apply_edge_incident(
        self,
        graph,
        source,
        target,
        travel_time_multiplier=3.0,
        congestion=0.95,
    ):
        """
        Directly apply an incident to an edge.
        """

        if graph is None:
            raise ValueError(
                "graph cannot be None"
            )

        source = str(source)
        target = str(target)

        for edge in graph.get_edges():

            if (
                str(edge.source.id) == source
                and str(edge.target.id) == target
            ):

                edge.travel_time *= float(
                    travel_time_multiplier
                )

                edge.congestion = min(
                    1.0,
                    max(
                        edge.congestion,
                        float(congestion),
                    ),
                )

                edge.blocked = True

                return True

        return False

    # =========================================================
    # DIRECT ROAD BLOCKAGE
    # =========================================================

    def apply_road_blockage(
        self,
        graph,
        source,
        target,
    ):
        """
        Directly block an edge.
        """

        if graph is None:
            raise ValueError(
                "graph cannot be None"
            )

        source = str(source)
        target = str(target)

        for edge in graph.get_edges():

            if (
                str(edge.source.id) == source
                and str(edge.target.id) == target
            ):

                edge.blocked = True

                return True

        return False


# =================================================================
# DYNAMIC TRAFFIC MODEL
# =================================================================


class DynamicTrafficModel:
    """
    Dynamic traffic model.

    Used by the dynamic-routing and benchmark experiments.

    The model modifies traffic conditions based on:

        - normal traffic
        - gradual congestion
        - incidents
        - moving congestion
        - random traffic variation

    Parameters
    ----------
    seed : int
        Random seed.

    base_congestion : float
        Normal/background congestion.

    congestion_factor : float
        Controls the relationship between congestion
        and travel time.

    branches_per_layer : int
        Number of branches in the layered benchmark
        network.
    """

    def __init__(
        self,
        seed=42,
        base_congestion=0.10,
        congestion_factor=5.0,
        branches_per_layer=3,
    ):
        self.seed = seed

        self.random = random.Random(
            seed
        )

        self.base_congestion = float(
            base_congestion
        )

        self.congestion_factor = float(
            congestion_factor
        )

        self.branches_per_layer = int(
            branches_per_layer
        )

        if self.branches_per_layer < 1:
            raise ValueError(
                "branches_per_layer must be "
                "at least 1."
            )

    # =========================================================
    # UPDATE EDGE
    # =========================================================

    def update_edge(
        self,
        edge,
        congestion,
    ):
        """
        Update congestion and travel time
        for one edge.
        """

        congestion = min(
            1.0,
            max(
                0.0,
                float(congestion),
            ),
        )

        edge.congestion = congestion

        # Preserve the base/free-flow travel time
        # when possible.
        #
        # The existing edge travel_time is treated
        # as the current value. To avoid repeated
        # runaway multiplication, use a stored
        # free-flow value.

        if not hasattr(
            edge,
            "_base_travel_time"
        ):
            edge._base_travel_time = (
                float(edge.travel_time)
            )

        edge.travel_time = (
            edge._base_travel_time
            * (
                1.0
                + self.congestion_factor
                * congestion
            )
        )

        return edge

    # =========================================================
    # NODE NUMBER
    # =========================================================

    def _get_node_number(
        self,
        node,
    ):
        """
        Extract numeric node number.

        Examples:

            N0 -> 0
            N25 -> 25
            25 -> 25
        """

        node_id = (
            node.id
            if hasattr(node, "id")
            else node
        )

        node_id = str(
            node_id
        )

        digits = ""

        for char in node_id:

            if char.isdigit():
                digits += char

        if not digits:
            return 0

        return int(digits)

    # =========================================================
    # LAYER NUMBER
    # =========================================================

    def _get_layer_number(
        self,
        node,
    ):
        """
        Determine the layer of a node in the
        layered benchmark network.

        N0 is treated as the source/depot.

        For branches_per_layer = 3:

            N0       -> layer 0

            N1-N3    -> layer 1

            N4-N6    -> layer 2

            N7-N9    -> layer 3
        """

        number = self._get_node_number(
            node
        )

        if number == 0:
            return 0

        return (
            (number - 1)
            // self.branches_per_layer
        ) + 1

    # =========================================================
    # EDGE TOUCHES LAYER
    # =========================================================

    def _edge_touches_layer(
        self,
        edge,
        layer,
    ):
        """
        Return True if an edge connects to
        the specified layer.
        """

        source_layer = (
            self._get_layer_number(
                edge.source
            )
        )

        target_layer = (
            self._get_layer_number(
                edge.target
            )
        )

        return (
            source_layer == layer
            or target_layer == layer
        )

    # =========================================================
    # NORMAL
    # =========================================================

    def normal(
        self,
        graph,
    ):
        """
        Restore normal traffic conditions.
        """

        for edge in graph.get_edges():

            edge.blocked = False

            self.update_edge(
                edge,
                self.base_congestion,
            )

        return graph

    # =========================================================
    # GRADUAL CONGESTION
    # =========================================================

    def gradual_congestion(
        self,
        graph,
        level,
    ):
        """
        Apply gradual congestion.

        level is expected in [0, 1].

        Example:

            0.25 -> moderate congestion
            0.50 -> heavy congestion
        """

        level = min(
            1.0,
            max(
                0.0,
                float(level),
            ),
        )

        congestion = min(
            1.0,
            self.base_congestion
            + level,
        )

        for edge in graph.get_edges():

            edge.blocked = False

            self.update_edge(
                edge,
                congestion,
            )

        return graph

    # =========================================================
    # INCIDENT
    # =========================================================

    def incident(
        self,
        graph,
        layer,
        incident_congestion=1.0,
    ):
        """
        Create a traffic incident affecting
        edges touching a specified layer.
        """

        layer = int(layer)

        for edge in graph.get_edges():

            if self._edge_touches_layer(
                edge,
                layer,
            ):

                self.update_edge(
                    edge,
                    incident_congestion,
                )

        return graph

    # =========================================================
    # MOVING CONGESTION
    # =========================================================

    def moving_congestion(
        self,
        graph,
        layer,
        congestion=0.90,
    ):
        """
        Apply a moving congestion zone to
        a specified layer.
        """

        layer = int(layer)

        congestion = min(
            1.0,
            max(
                0.0,
                float(congestion),
            ),
        )

        for edge in graph.get_edges():

            if self._edge_touches_layer(
                edge,
                layer,
            ):

                self.update_edge(
                    edge,
                    congestion,
                )

        return graph

    # =========================================================
    # RANDOM TRAFFIC
    # =========================================================

    def random_traffic(
        self,
        graph,
        variation=0.20,
    ):
        """
        Apply random traffic variation.

        variation=0.20 means approximately
        +/- 20% around the base congestion.
        """

        variation = abs(
            float(variation)
        )

        for edge in graph.get_edges():

            random_factor = (
                self.random.uniform(
                    -variation,
                    variation,
                )
            )

            congestion = (
                self.base_congestion
                * (
                    1.0
                    + random_factor
                )
            )

            congestion = min(
                1.0,
                max(
                    0.0,
                    congestion,
                ),
            )

            self.update_edge(
                edge,
                congestion,
            )

        return graph
class TrafficScenario:
    """
    Traffic scenario representation.

    Supports:

    - Normal traffic
    - Peak traffic
    - Congested traffic
    - Road incidents
    - Road blockages
    - Custom/dynamic scenarios
    - Multiple edge-specific updates

    Example:

        scenario = TrafficScenario("incident")

        scenario.add_update(
            "N0",
            "N1",
            travel_time=30,
            congestion=0.9
        )

        scenario.apply(graph)
    """

    # =========================================================
    # SCENARIO CONSTANTS
    # =========================================================

    NORMAL = "normal"
    PEAK = "peak"
    CONGESTED = "congested"
    INCIDENT = "incident"
    ROAD_BLOCKAGE = "road_blockage"

    VALID_TYPES = {
        NORMAL,
        PEAK,
        CONGESTED,
        INCIDENT,
        ROAD_BLOCKAGE,
    }

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        scenario_type,
        congestion_multiplier=1.0,
        travel_time_multiplier=1.0,
        source=None,
        target=None,
        blocked=False,
        congestion=None,
    ):
        """
        Create a traffic scenario.

        Custom scenario names are allowed.

        Example:

            TrafficScenario("test")
            TrafficScenario("incident")
            TrafficScenario("peak")
        """

        self.scenario_type = str(
            scenario_type
        ).lower()

        # Compatibility with existing tests.
        self.name = self.scenario_type

        self.congestion_multiplier = float(
            congestion_multiplier
        )

        self.travel_time_multiplier = float(
            travel_time_multiplier
        )

        self.source = (
            None
            if source is None
            else str(source)
        )

        self.target = (
            None
            if target is None
            else str(target)
        )

        self.blocked = bool(blocked)

        self.congestion = (
            None
            if congestion is None
            else float(congestion)
        )

        # Stores dynamic edge updates.
        self.updates = []

    # =========================================================
    # ADD UPDATE
    # =========================================================

    def add_update(
        self,
        source,
        target,
        travel_time=None,
        congestion=None,
        distance=None,
        blocked=None,
    ):
        """
        Add a dynamic traffic update for an edge.

        Parameters
        ----------
        source : str
            Source node ID.

        target : str
            Target node ID.

        travel_time : float, optional
            New travel time.

        congestion : float, optional
            New congestion value.

        distance : float, optional
            New distance.

        blocked : bool, optional
            Whether the road is blocked.

        Example
        -------

            scenario.add_update(
                "N0",
                "N1",
                travel_time=30,
                congestion=0.9
            )
        """

        update = {
            "source": str(source),
            "target": str(target),

            "travel_time": (
                None
                if travel_time is None
                else float(travel_time)
            ),

            "congestion": (
                None
                if congestion is None
                else float(congestion)
            ),

            "distance": (
                None
                if distance is None
                else float(distance)
            ),

            "blocked": blocked,
        }

        self.updates.append(update)

        return self

    # =========================================================
    # GET UPDATE
    # =========================================================

    def get_update(
        self,
        source,
        target,
    ):
        """
        Return the update associated with
        a specific directed edge.

        Returns
        -------
        dict or None
        """

        source = str(source)
        target = str(target)

        for update in self.updates:

            if (
                update["source"] == source
                and update["target"] == target
            ):
                return update

        return None

    # =========================================================
    # REMOVE UPDATE
    # =========================================================

    def remove_update(
        self,
        source,
        target,
    ):
        """
        Remove an update for a specific edge.

        Returns
        -------
        bool
            True if an update was removed.
            False otherwise.
        """

        source = str(source)
        target = str(target)

        for index, update in enumerate(
            self.updates
        ):

            if (
                update["source"] == source
                and update["target"] == target
            ):

                self.updates.pop(index)

                return True

        return False

    # =========================================================
    # CLEAR UPDATES
    # =========================================================

    def clear_updates(self):
        """
        Remove all dynamic traffic updates.
        """

        self.updates.clear()

    # =========================================================
    # APPLY SCENARIO
    # =========================================================

    def apply(self, graph):
        """
        Apply the scenario directly to a graph.

        The supplied graph is modified and returned.
        """

        if graph is None:
            raise ValueError(
                "graph cannot be None"
            )

        # =====================================================
        # DYNAMIC UPDATES
        # =====================================================

        if self.updates:

            self._apply_updates(graph)

            return graph

        # =====================================================
        # NORMAL
        # =====================================================

        if self.scenario_type == self.NORMAL:

            for edge in graph.get_edges():

                edge.blocked = False

            return graph

        # =====================================================
        # PEAK
        # =====================================================

        if self.scenario_type == self.PEAK:

            for edge in graph.get_edges():

                edge.congestion = min(
                    1.0,
                    edge.congestion + 0.20,
                )

                edge.travel_time *= 1.20

            return graph

        # =====================================================
        # CONGESTED
        # =====================================================

        if self.scenario_type == self.CONGESTED:

            for edge in graph.get_edges():

                edge.congestion = min(
                    1.0,
                    edge.congestion + 0.40,
                )

                edge.travel_time *= 1.40

            return graph

        # =====================================================
        # INCIDENT
        # =====================================================

        if self.scenario_type == self.INCIDENT:

            return self._apply_edge_scenario(
                graph
            )

        # =====================================================
        # ROAD BLOCKAGE
        # =====================================================

        if self.scenario_type == self.ROAD_BLOCKAGE:

            return self._apply_edge_scenario(
                graph
            )

        # =====================================================
        # CUSTOM SCENARIO
        # =====================================================

        # Custom scenarios such as:
        #
        #     TrafficScenario("test")
        #
        # are allowed.
        #
        # If they contain updates, those updates
        # were already applied above.
        #
        # If they contain no updates, do nothing.

        return graph

    # =========================================================
    # APPLY DYNAMIC UPDATES
    # =========================================================

    def _apply_updates(self, graph):
        """
        Apply every update stored in self.updates.
        """

        for update in self.updates:

            source = update["source"]
            target = update["target"]

            found = False

            for edge in graph.get_edges():

                source_id = str(
                    edge.source.id
                )

                target_id = str(
                    edge.target.id
                )

                if (
                    source_id == source
                    and target_id == target
                ):

                    # -----------------------------------------
                    # TRAVEL TIME
                    # -----------------------------------------

                    if (
                        update["travel_time"]
                        is not None
                    ):

                        edge.travel_time = (
                            update["travel_time"]
                        )

                    # -----------------------------------------
                    # CONGESTION
                    # -----------------------------------------

                    if (
                        update["congestion"]
                        is not None
                    ):

                        edge.congestion = min(
                            1.0,
                            max(
                                0.0,
                                update["congestion"],
                            ),
                        )

                    # -----------------------------------------
                    # DISTANCE
                    # -----------------------------------------

                    if (
                        update["distance"]
                        is not None
                    ):

                        edge.distance = (
                            update["distance"]
                        )

                    # -----------------------------------------
                    # BLOCKED
                    # -----------------------------------------

                    if (
                        update["blocked"]
                        is not None
                    ):

                        edge.blocked = bool(
                            update["blocked"]
                        )

                    found = True

                    break

            if not found:

                raise ValueError(
                    f"Edge {source} -> "
                    f"{target} not found."
                )

        return graph

    # =========================================================
    # APPLY EDGE SCENARIO
    # =========================================================

    def _apply_edge_scenario(self, graph):
        """
        Apply an incident or road blockage to
        one specific directed edge.
        """

        if self.source is None:

            raise ValueError(
                "source is required for "
                "an edge scenario."
            )

        if self.target is None:

            raise ValueError(
                "target is required for "
                "an edge scenario."
            )

        found = False

        for edge in graph.get_edges():

            source_id = str(
                edge.source.id
            )

            target_id = str(
                edge.target.id
            )

            if (
                source_id == self.source
                and target_id == self.target
            ):

                # =================================================
                # INCIDENT
                # =================================================

                if (
                    self.scenario_type
                    == self.INCIDENT
                ):

                    edge.travel_time *= (
                        self.travel_time_multiplier
                    )

                    if self.congestion is not None:

                        edge.congestion = min(
                            1.0,
                            max(
                                edge.congestion,
                                self.congestion,
                            ),
                        )

                    else:

                        edge.congestion = min(
                            1.0,
                            max(
                                edge.congestion,
                                0.95,
                            ),
                        )

                    if self.blocked:

                        edge.blocked = True

                # =================================================
                # ROAD BLOCKAGE
                # =================================================

                elif (
                    self.scenario_type
                    == self.ROAD_BLOCKAGE
                ):

                    edge.blocked = True

                found = True

                break

        if not found:

            raise ValueError(
                f"Edge {self.source} -> "
                f"{self.target} not found."
            )

        return graph

    # =========================================================
    # UPDATE ALIAS
    # =========================================================

    def update(self, graph):
        """
        Alias for apply().
        """

        return self.apply(graph)

    # =========================================================
    # REPRESENTATION
    # =========================================================

    def __repr__(self):

        return (
            f"TrafficScenario("
            f"name='{self.name}', "
            f"type='{self.scenario_type}', "
            f"updates={len(self.updates)}, "
            f"source={self.source}, "
            f"target={self.target}, "
            f"blocked={self.blocked}"
            f")"
        )


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

Scenario = TrafficScenario


# =============================================================
# CREATE SCENARIO
# =============================================================

def create_scenario(
    scenario_type,
    source=None,
    target=None,
    travel_time_multiplier=1.0,
    congestion_multiplier=1.0,
    congestion=None,
    blocked=False,
):
    """
    Create a TrafficScenario.

    Examples:

        create_scenario(
            TrafficScenario.NORMAL
        )

        create_scenario(
            TrafficScenario.PEAK
        )

        create_scenario(
            TrafficScenario.CONGESTED
        )

        create_scenario(
            TrafficScenario.INCIDENT,
            source="N0",
            target="N1",
            travel_time_multiplier=3.0,
            congestion=0.95,
        )

        create_scenario(
            TrafficScenario.ROAD_BLOCKAGE,
            source="N0",
            target="N1",
        )
    """

    scenario_type = str(
        scenario_type
    ).lower()

    # =========================================================
    # NORMAL
    # =========================================================

    if scenario_type == TrafficScenario.NORMAL:

        return TrafficScenario(
            scenario_type=TrafficScenario.NORMAL,
            congestion_multiplier=1.0,
            travel_time_multiplier=1.0,
        )

    # =========================================================
    # PEAK
    # =========================================================

    if scenario_type == TrafficScenario.PEAK:

        return TrafficScenario(
            scenario_type=TrafficScenario.PEAK,
            congestion_multiplier=1.20,
            travel_time_multiplier=1.20,
        )

    # =========================================================
    # CONGESTED
    # =========================================================

    if (
        scenario_type
        == TrafficScenario.CONGESTED
    ):

        return TrafficScenario(
            scenario_type=TrafficScenario.CONGESTED,
            congestion_multiplier=1.40,
            travel_time_multiplier=1.40,
        )

    # =========================================================
    # INCIDENT
    # =========================================================

    if (
        scenario_type
        == TrafficScenario.INCIDENT
    ):

        if source is None or target is None:

            raise ValueError(
                "source and target are required "
                "for an incident scenario."
            )

        if congestion is None:

            congestion = 0.95

        return TrafficScenario(
            scenario_type=TrafficScenario.INCIDENT,
            source=source,
            target=target,
            travel_time_multiplier=(
                travel_time_multiplier
            ),
            congestion=congestion,
            blocked=blocked,
        )

    # =========================================================
    # ROAD BLOCKAGE
    # =========================================================

    if (
        scenario_type
        == TrafficScenario.ROAD_BLOCKAGE
    ):

        if source is None or target is None:

            raise ValueError(
                "source and target are required "
                "for a road blockage scenario."
            )

        return TrafficScenario(
            scenario_type=(
                TrafficScenario.ROAD_BLOCKAGE
            ),
            source=source,
            target=target,
            travel_time_multiplier=1.0,
            congestion_multiplier=1.0,
            blocked=True,
        )

    # =========================================================
    # CUSTOM SCENARIO
    # =========================================================

    return TrafficScenario(
        scenario_type=scenario_type,
        source=source,
        target=target,
        travel_time_multiplier=(
            travel_time_multiplier
        ),
        congestion_multiplier=(
            congestion_multiplier
        ),
        congestion=congestion,
        blocked=blocked,
    )
class RouteUtils:
    """
    Utilities for converting QPSO decisions into
    valid graph routes.
    """

    @staticmethod
    def edge_score_to_index(score, edge_count):
        """
        Convert a continuous QPSO value in [0, 1]
        into an edge index.
        """

        if edge_count <= 0:
            raise ValueError(
                "edge_count must be positive."
            )

        score = max(
            0.0,
            min(1.0, float(score))
        )

        index = int(
            score * edge_count
        )

        if index >= edge_count:
            index = edge_count - 1

        return index

    @staticmethod
    def build_route(
        graph,
        start,
        destination,
        decisions,
    ):
        """
        Convert continuous QPSO decisions into a
        feasible route.

        At every node, the decision selects one
        outgoing edge.

        The route terminates when destination is
        reached or when a cycle is detected.
        """

        if not decisions:
            return None

        current = graph.get_node(start)
        target = graph.get_node(destination)

        route = [current]
        visited = {current.id}

        decision_index = 0

        while current.id != target.id:

            neighbors = [
                edge
                for edge in graph.get_neighbors(current)
                if not edge.blocked
            ]

            if not neighbors:
                return None

            score = decisions[
                decision_index
                % len(decisions)
            ]

            edge_index = (
                RouteUtils.edge_score_to_index(
                    score,
                    len(neighbors)
                )
            )

            selected_edge = neighbors[
                edge_index
            ]

            current = selected_edge.target

            if current.id in visited:
                return None

            route.append(current)
            visited.add(current.id)

            decision_index += 1

            # Safety limit prevents an invalid
            # network from producing an infinite loop.
            if decision_index > graph.number_of_nodes():
                return None

        return route

    @staticmethod
    def route_is_valid(
        graph,
        route,
        start,
        destination,
    ):
        """
        Verify that a route is feasible.
        """

        if route is None or len(route) < 2:
            return False

        if route[0].id != str(start):
            return False

        if route[-1].id != str(destination):
            return False

        for source, target in zip(
            route[:-1],
            route[1:]
        ):
            valid_edge = False

            for edge in graph.get_neighbors(
                source
            ):
                if (
                    edge.target.id == target.id
                    and not edge.blocked
                ):
                    valid_edge = True
                    break

            if not valid_edge:
                return False

        return True

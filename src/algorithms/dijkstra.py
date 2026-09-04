import heapq
from math import inf


class Dijkstra:
    """
    Dijkstra's shortest-path algorithm.

    Finds a minimum-travel-time route while ignoring blocked edges.

    Parameters
    ----------
    graph:
        Directed weighted graph.

    fitness_function:
        Optional fitness function kept for compatibility with the
        benchmarking framework. Dijkstra primarily uses travel time
        for path selection.
    """

    def __init__(self, graph, fitness_function=None):
        if graph is None:
            raise ValueError("Graph cannot be None.")

        self.graph = graph
        self.fitness_function = fitness_function

    def optimize(self, start, destination):
        """
        Find the shortest travel-time route from start to destination.

        Returns
        -------
        dict
            {
                "route": list[Node],
                "travel_time": float,
                "distance": float,
                "congestion": float,
                "fitness": float
            }
        """

        start = str(start)
        destination = str(destination)

        # ---------------------------------------------------------
        # Validate start and destination
        # ---------------------------------------------------------

        try:
            self.graph.get_node(start)
            self.graph.get_node(destination)
        except KeyError as exc:
            raise ValueError(
                f"Unknown node in route: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------

        distances = {
            start: 0.0
        }

        previous = {
            start: None
        }

        priority_queue = [
            (0.0, start)
        ]

        visited = set()

        # ---------------------------------------------------------
        # Dijkstra
        # ---------------------------------------------------------

        while priority_queue:

            current_distance, current_id = heapq.heappop(
                priority_queue
            )

            # Ignore stale queue entries
            if current_distance > distances.get(
                current_id,
                inf
            ):
                continue

            # Ignore already processed nodes
            if current_id in visited:
                continue

            visited.add(current_id)

            # Destination reached
            if current_id == destination:
                break

            current_node = self.graph.get_node(
                current_id
            )

            # -----------------------------------------------------
            # Explore outgoing edges
            # -----------------------------------------------------

            for edge in self.graph.get_neighbors(
                current_node
            ):

                # IMPORTANT:
                # Blocked roads cannot be used.
                if edge.blocked:
                    continue

                neighbor_id = edge.target.id

                new_distance = (
                    current_distance
                    + float(edge.travel_time)
                )

                old_distance = distances.get(
                    neighbor_id,
                    inf
                )

                if new_distance < old_distance:

                    distances[neighbor_id] = new_distance

                    previous[neighbor_id] = current_id

                    heapq.heappush(
                        priority_queue,
                        (
                            new_distance,
                            neighbor_id
                        )
                    )

        # ---------------------------------------------------------
        # Check whether destination was reached
        # ---------------------------------------------------------

        if destination not in distances:
            raise ValueError(
                f"No route exists from "
                f"{start} to {destination}."
            )

        # ---------------------------------------------------------
        # Reconstruct route
        # ---------------------------------------------------------

        route_ids = []

        current_id = destination

        while current_id is not None:

            route_ids.append(current_id)

            current_id = previous.get(
                current_id
            )

        route_ids.reverse()

        if not route_ids or route_ids[0] != start:
            raise ValueError(
                f"Unable to reconstruct route from "
                f"{start} to {destination}."
            )

        route = [
            self.graph.get_node(node_id)
            for node_id in route_ids
        ]

        # ---------------------------------------------------------
        # Calculate route metrics
        # ---------------------------------------------------------

        travel_time = 0.0
        distance = 0.0
        congestion_values = []

        for source, target in zip(
            route[:-1],
            route[1:]
        ):

            selected_edge = None

            for edge in self.graph.get_neighbors(
                source
            ):

                if (
                    edge.target.id == target.id
                    and not edge.blocked
                ):
                    selected_edge = edge
                    break

            if selected_edge is None:
                raise ValueError(
                    "Route contains invalid edge: "
                    f"{source.id} -> {target.id}"
                )

            travel_time += float(
                selected_edge.travel_time
            )

            distance += float(
                selected_edge.distance
            )

            congestion_values.append(
                float(selected_edge.congestion)
            )

        # ---------------------------------------------------------
        # Average congestion
        # ---------------------------------------------------------

        if congestion_values:
            congestion = (
                sum(congestion_values)
                / len(congestion_values)
            )
        else:
            congestion = 0.0

        # ---------------------------------------------------------
        # Fitness
        # ---------------------------------------------------------

        if self.fitness_function is not None:

            fitness = self.fitness_function.calculate(
                travel_time=travel_time,
                distance=distance,
                congestion=congestion,
            )

        else:

            # Backward-compatible default:
            # Dijkstra minimizes travel time.
            fitness = travel_time

        # ---------------------------------------------------------
        # Result
        # ---------------------------------------------------------

        return {
            "route": route,
            "travel_time": travel_time,
            "distance": distance,
            "congestion": congestion,
            "fitness": fitness,
        }
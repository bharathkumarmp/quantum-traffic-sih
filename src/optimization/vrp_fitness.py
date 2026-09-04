from src.algorithms.dijkstra import Dijkstra


class VRPFitnessFunction:
    """
    Fitness function for the Vehicle Routing Problem (VRP).

    Objective:
        Minimize total travel time,
        total travel distance,
        and average congestion.

    Lower fitness is better.

    Fitness:
        F =
            time_weight * travel_time
            + distance_weight * distance
            + congestion_weight * average_congestion
            + penalties

    A shortest-path cache is used because VRP optimization repeatedly
    requests the same node-to-node paths.
    """

    def __init__(
        self,
        time_weight=0.5,
        distance_weight=0.3,
        congestion_weight=0.2,
        invalid_penalty=1_000_000.0,
        capacity_penalty=10_000.0,
        unreachable_penalty=100_000.0,
    ):
        total_weight = (
            time_weight
            + distance_weight
            + congestion_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "Fitness weights must have positive sum."
            )

        self.time_weight = (
            float(time_weight) / total_weight
        )

        self.distance_weight = (
            float(distance_weight) / total_weight
        )

        self.congestion_weight = (
            float(congestion_weight) / total_weight
        )

        self.invalid_penalty = float(
            invalid_penalty
        )

        self.capacity_penalty = float(
            capacity_penalty
        )

        self.unreachable_penalty = float(
            unreachable_penalty
        )

        # =====================================================
        # SHORTEST-PATH CACHE
        # =====================================================
        #
        # Key:
        #     (graph identity, start node, destination node)
        #
        # Graph identity is important because different traffic
        # scenarios can create different graph objects with
        # different edge weights.
        #
        self._shortest_path_cache = {}

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

    # =========================================================
    # SHORTEST PATH
    # =========================================================

    def _shortest_path(
        self,
        graph,
        start,
        destination,
    ):
        """
        Find the shortest path using Dijkstra.

        Results are cached because VRP optimization repeatedly
        requests the same node-to-node shortest paths.

        Returns:
            Dictionary containing route and metrics.

        Returns:
            None if the destination is unreachable.
        """

        start = str(start)
        destination = str(destination)

        # -----------------------------------------------------
        # Same-node path
        # -----------------------------------------------------

        if start == destination:

            node = graph.get_node(start)

            return {
                "route": [node],
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "fitness": 0.0,
            }

        # -----------------------------------------------------
        # Cache key
        # -----------------------------------------------------

        cache_key = (
            id(graph),
            start,
            destination,
        )

        # -----------------------------------------------------
        # Cache lookup
        # -----------------------------------------------------

        if cache_key in self._shortest_path_cache:

            self._cache_hits += 1

            return self._shortest_path_cache[
                cache_key
            ]

        self._cache_misses += 1

        # -----------------------------------------------------
        # Dijkstra
        # -----------------------------------------------------

        try:

            dijkstra = Dijkstra(graph)

            result = dijkstra.optimize(
                start,
                destination,
            )

        except (ValueError, KeyError):

            result = None

        # -----------------------------------------------------
        # Store result
        # -----------------------------------------------------

        self._shortest_path_cache[
            cache_key
        ] = result

        return result

    # =========================================================
    # PATH CONGESTION
    # =========================================================

    def _path_congestion(
        self,
        graph,
        route,
    ):
        """
        Calculate average congestion along a physical path.

        `route` contains Node objects.
        """

        if route is None or len(route) < 2:
            return 0.0

        congestion_values = []

        for index in range(
            len(route) - 1
        ):

            source = route[index]
            target = route[index + 1]

            source_id = str(source.id)
            target_id = str(target.id)

            try:

                neighbors = graph.get_neighbors(
                    source
                )

            except (AttributeError, KeyError):

                continue

            for edge in neighbors:

                if (
                    str(edge.target.id)
                    == target_id
                ):

                    congestion_values.append(
                        float(edge.congestion)
                    )

                    break

        if not congestion_values:
            return 0.0

        return (
            sum(congestion_values)
            / len(congestion_values)
        )

    # =========================================================
    # VEHICLE ROUTE EVALUATION
    # =========================================================

    def evaluate_vehicle_route(
        self,
        problem,
        vehicle,
        customer_ids,
    ):
        """
        Evaluate one vehicle route.

        Route structure:

            vehicle.start_node
                    ↓
               customer 1
                    ↓
               customer 2
                    ↓
                  ...
                    ↓
               vehicle.end_node

        `route` in the returned result is the physical route
        containing Node objects.

        `customers` contains the customer IDs.
        """

        # -----------------------------------------------------
        # Vehicle lookup
        # -----------------------------------------------------

        if isinstance(vehicle, str):

            try:

                vehicle = problem.get_vehicle(
                    vehicle
                )

            except KeyError:

                return {
                    "valid": False,
                    "fitness": self.invalid_penalty,
                    "travel_time": 0.0,
                    "distance": 0.0,
                    "congestion": 0.0,
                    "penalty": self.invalid_penalty,
                    "route": [],
                    "customers": [],
                    "demand": 0.0,
                }

        elif not hasattr(vehicle, "id"):

            try:

                vehicle = problem.get_vehicle(
                    vehicle
                )

            except (KeyError, TypeError):

                return {
                    "valid": False,
                    "fitness": self.invalid_penalty,
                    "travel_time": 0.0,
                    "distance": 0.0,
                    "congestion": 0.0,
                    "penalty": self.invalid_penalty,
                    "route": [],
                    "customers": [],
                    "demand": 0.0,
                }

        # -----------------------------------------------------
        # Normalize customer route
        # -----------------------------------------------------

        if customer_ids is None:
            customer_ids = []

        customer_ids = [
            str(customer_id)
            for customer_id in customer_ids
        ]

        # -----------------------------------------------------
        # Empty route
        # -----------------------------------------------------

        if not customer_ids:

            return {
                "valid": True,
                "fitness": 0.0,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": 0.0,
                "route": [],
                "customers": [],
                "demand": 0.0,
            }

        # -----------------------------------------------------
        # Customer validation
        # -----------------------------------------------------

        try:

            customers = [
                problem.get_customer(
                    customer_id
                )
                for customer_id in customer_ids
            ]

        except KeyError:

            return {
                "valid": False,
                "fitness": self.invalid_penalty,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": self.invalid_penalty,
                "route": [],
                "customers": customer_ids,
                "demand": 0.0,
            }

        # -----------------------------------------------------
        # Capacity
        # -----------------------------------------------------

        total_demand = sum(
            customer.demand
            for customer in customers
        )

        capacity_penalty = 0.0

        if total_demand > vehicle.capacity:

            capacity_penalty = (
                self.capacity_penalty
                * (
                    total_demand
                    - vehicle.capacity
                )
            )

        # -----------------------------------------------------
        # Build physical node sequence
        # -----------------------------------------------------

        node_sequence = [
            str(vehicle.start_node)
        ]

        for customer in customers:

            node_sequence.append(
                str(customer.node_id)
            )

        node_sequence.append(
            str(vehicle.end_node)
        )

        # -----------------------------------------------------
        # Evaluate consecutive node pairs
        # -----------------------------------------------------

        total_travel_time = 0.0
        total_distance = 0.0

        congestion_values = []

        complete_route = []

        unreachable = False

        for index in range(
            len(node_sequence) - 1
        ):

            start_node = node_sequence[
                index
            ]

            end_node = node_sequence[
                index + 1
            ]

            path_result = self._shortest_path(
                problem.graph,
                start_node,
                end_node,
            )

            # -------------------------------------------------
            # Unreachable
            # -------------------------------------------------

            if path_result is None:

                unreachable = True

                break

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            total_travel_time += float(
                path_result.get(
                    "travel_time",
                    0.0,
                )
            )

            total_distance += float(
                path_result.get(
                    "distance",
                    0.0,
                )
            )

            # -------------------------------------------------
            # Physical path
            # -------------------------------------------------

            path = path_result.get(
                "route",
                [],
            )

            if path:

                if not complete_route:

                    complete_route.extend(
                        path
                    )

                else:

                    # Avoid duplicating the connecting node.
                    complete_route.extend(
                        path[1:]
                    )

            # -------------------------------------------------
            # Congestion
            # -------------------------------------------------

            path_congestion = (
                path_result.get(
                    "congestion",
                    None,
                )
            )

            if path_congestion is not None:

                congestion_values.append(
                    float(path_congestion)
                )

        # =====================================================
        # UNREACHABLE ROUTE
        # =====================================================

        if unreachable:

            return {
                "valid": False,
                "fitness": (
                    self.unreachable_penalty
                    + capacity_penalty
                ),
                "travel_time": total_travel_time,
                "distance": total_distance,
                "congestion": 0.0,
                "penalty": (
                    self.unreachable_penalty
                    + capacity_penalty
                ),
                "route": complete_route,
                "customers": customer_ids,
                "demand": total_demand,
                "capacity": vehicle.capacity,
            }

        # =====================================================
        # AVERAGE CONGESTION
        # =====================================================

        if congestion_values:

            average_congestion = (
                sum(congestion_values)
                / len(congestion_values)
            )

        else:

            average_congestion = (
                self._path_congestion(
                    problem.graph,
                    complete_route,
                )
            )

        # =====================================================
        # FITNESS
        # =====================================================

        fitness = (
            self.time_weight
            * total_travel_time
        )

        fitness += (
            self.distance_weight
            * total_distance
        )

        fitness += (
            self.congestion_weight
            * min(
                1.0,
                average_congestion,
            )
        )

        fitness += capacity_penalty

        # =====================================================
        # VALIDITY
        # =====================================================

        valid = (
            total_demand
            <= vehicle.capacity
        )

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "valid": valid,
            "fitness": fitness,
            "travel_time": total_travel_time,
            "distance": total_distance,
            "congestion": average_congestion,
            "penalty": capacity_penalty,
            "route": complete_route,
            "customers": customer_ids,
            "demand": total_demand,
            "capacity": vehicle.capacity,
            "path": complete_route,
        }

    # =========================================================
    # COMPLETE VRP EVALUATION
    # =========================================================

    def evaluate(
        self,
        problem,
        solution,
    ):
        """
        Evaluate a complete VRP solution.

        Expected solution format:

            {
                "V1": ["C1", "C5"],
                "V2": ["C2", "C3"],
                ...
            }

        Every vehicle must appear exactly once.

        Every customer must appear exactly once.
        """

        if not isinstance(
            solution,
            dict,
        ):

            return {
                "valid": False,
                "fitness": self.invalid_penalty,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": self.invalid_penalty,
                "vehicle_results": {},
            }

        # =====================================================
        # VEHICLE ID VALIDATION
        # =====================================================

        expected_vehicle_ids = {
            str(vehicle.id)
            for vehicle in problem.vehicles
        }

        solution_vehicle_ids = {
            str(vehicle_id)
            for vehicle_id in solution.keys()
        }

        if (
            solution_vehicle_ids
            != expected_vehicle_ids
        ):

            return {
                "valid": False,
                "fitness": self.invalid_penalty,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": self.invalid_penalty,
                "vehicle_results": {},
            }

        # =====================================================
        # CUSTOMER VALIDATION
        # =====================================================

        expected_customers = {
            str(customer.id)
            for customer in problem.customers
        }

        assigned_customers = []

        for route in solution.values():

            if not isinstance(
                route,
                list,
            ):

                return {
                    "valid": False,
                    "fitness": self.invalid_penalty,
                    "travel_time": 0.0,
                    "distance": 0.0,
                    "congestion": 0.0,
                    "penalty": self.invalid_penalty,
                    "vehicle_results": {},
                }

            assigned_customers.extend(
                str(customer_id)
                for customer_id in route
            )

        assigned_customer_set = set(
            assigned_customers
        )

        # -----------------------------------------------------
        # Duplicate customers
        # -----------------------------------------------------

        if len(assigned_customers) != len(
            assigned_customer_set
        ):

            return {
                "valid": False,
                "fitness": self.invalid_penalty,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": self.invalid_penalty,
                "vehicle_results": {},
            }

        # -----------------------------------------------------
        # Missing or extra customers
        # -----------------------------------------------------

        if (
            assigned_customer_set
            != expected_customers
        ):

            return {
                "valid": False,
                "fitness": self.invalid_penalty,
                "travel_time": 0.0,
                "distance": 0.0,
                "congestion": 0.0,
                "penalty": self.invalid_penalty,
                "vehicle_results": {},
            }

        # =====================================================
        # EVALUATE VEHICLES
        # =====================================================

        vehicle_results = {}

        total_travel_time = 0.0
        total_distance = 0.0

        congestion_values = []

        total_penalty = 0.0

        all_valid = True

        for vehicle in problem.vehicles:

            vehicle_id = str(
                vehicle.id
            )

            route = solution.get(
                vehicle_id,
                [],
            )

            vehicle_result = (
                self.evaluate_vehicle_route(
                    problem,
                    vehicle,
                    route,
                )
            )

            vehicle_results[
                vehicle_id
            ] = vehicle_result

            total_travel_time += float(
                vehicle_result.get(
                    "travel_time",
                    0.0,
                )
            )

            total_distance += float(
                vehicle_result.get(
                    "distance",
                    0.0,
                )
            )

            congestion = vehicle_result.get(
                "congestion",
                0.0,
            )

            if route:

                congestion_values.append(
                    float(congestion)
                )

            total_penalty += float(
                vehicle_result.get(
                    "penalty",
                    0.0,
                )
            )

            if not vehicle_result.get(
                "valid",
                False,
            ):

                all_valid = False

        # =====================================================
        # AVERAGE CONGESTION
        # =====================================================

        if congestion_values:

            average_congestion = (
                sum(congestion_values)
                / len(congestion_values)
            )

        else:

            average_congestion = 0.0

        # =====================================================
        # COMPLETE FITNESS
        # =====================================================

        fitness = (
            self.time_weight
            * total_travel_time
        )

        fitness += (
            self.distance_weight
            * total_distance
        )

        fitness += (
            self.congestion_weight
            * min(
                1.0,
                average_congestion,
            )
        )

        fitness += total_penalty

        # -----------------------------------------------------
        # Invalid solution penalty
        # -----------------------------------------------------

        if not all_valid:

            fitness += self.invalid_penalty

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "valid": all_valid,
            "fitness": fitness,
            "travel_time": total_travel_time,
            "distance": total_distance,
            "congestion": average_congestion,
            "penalty": total_penalty,
            "vehicle_results": vehicle_results,
            "solution": solution,
        }

    # =========================================================
    # CACHE INFORMATION
    # =========================================================

    def cache_info(self):
        """
        Return shortest-path cache statistics.
        """

        total_requests = (
            self._cache_hits
            + self._cache_misses
        )

        if total_requests > 0:

            hit_rate = (
                self._cache_hits
                / total_requests
            )

        else:

            hit_rate = 0.0

        return {
            "entries": len(
                self._shortest_path_cache
            ),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "requests": total_requests,
            "hit_rate": hit_rate,
        }

    # =========================================================
    # CLEAR CACHE
    # =========================================================

    def clear_cache(self):
        """
        Clear all shortest-path cache entries
        and reset cache statistics.
        """

        self._shortest_path_cache.clear()

        self._cache_hits = 0
        self._cache_misses = 0
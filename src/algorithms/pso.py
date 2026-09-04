import random

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.route_utils import RouteUtils
from src.optimization.fitness import FitnessFunction


class PSO:
    """
    Classical Particle Swarm Optimization (PSO)
    for traffic-route optimization.

    The algorithm searches a continuous position space
    and converts each position into a discrete graph route.

    Two initialization modes are supported:

    1. use_dijkstra_seed=True
       - First particle is initialized using Dijkstra.
       - Useful for production and scalability experiments.
       - Avoids recursive random-route initialization.

    2. use_dijkstra_seed=False
       - Population is generated using randomized feasible routes.
       - Useful for research and convergence experiments.

    Lower fitness is better.
    """

    def __init__(
        self,
        graph,
        particles=20,
        iterations=50,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=None,
        fitness_function=None,
        use_dijkstra_seed=True,
        initialization_attempts=100,
    ):
        if particles < 2:
            raise ValueError(
                "particles must be at least 2."
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be positive."
            )

        if inertia < 0:
            raise ValueError(
                "inertia must be non-negative."
            )

        if cognitive < 0:
            raise ValueError(
                "cognitive coefficient must be non-negative."
            )

        if social < 0:
            raise ValueError(
                "social coefficient must be non-negative."
            )

        if initialization_attempts < 1:
            raise ValueError(
                "initialization_attempts must be positive."
            )

        self.graph = graph

        self.particle_count = int(
            particles
        )

        self.iterations = int(
            iterations
        )

        self.inertia = float(
            inertia
        )

        self.cognitive = float(
            cognitive
        )

        self.social = float(
            social
        )

        self.random = random.Random(
            seed
        )

        self.fitness_function = (
            fitness_function
            if fitness_function is not None
            else FitnessFunction()
        )

        self.use_dijkstra_seed = bool(
            use_dijkstra_seed
        )

        self.initialization_attempts = int(
            initialization_attempts
        )

        self.edges = graph.get_edges()

        if not self.edges:
            raise ValueError(
                "Graph must contain at least one edge."
            )

        # A simple route can contain at most
        # number_of_nodes - 1 edges.
        self.dimension = max(
            1,
            graph.number_of_nodes() - 1,
        )

    # =========================================================
    # POSITION INITIALIZATION
    # =========================================================

    def _random_position(self):
        """
        Generate a random continuous position.
        """

        return [
            self.random.random()
            for _ in range(
                self.dimension
            )
        ]

    # =========================================================
    # VELOCITY INITIALIZATION
    # =========================================================

    def _random_velocity(self):
        """
        Generate an initial PSO velocity.
        """

        return [
            self.random.uniform(
                -0.5,
                0.5,
            )
            for _ in range(
                self.dimension
            )
        ]

    # =========================================================
    # POSITION -> ROUTE
    # =========================================================

    def _position_to_route(
        self,
        position,
        start,
        destination,
    ):
        """
        Convert a continuous PSO position
        into a discrete graph route.
        """

        return RouteUtils.build_route(
            self.graph,
            start,
            destination,
            position,
        )

    # =========================================================
    # ROUTE -> POSITION
    # =========================================================

    def _route_to_position(self, route):
        """
        Convert a valid graph route into
        a continuous PSO position.

        Each selected outgoing edge is encoded
        inside its corresponding [0, 1] interval.
        """

        position = [
            self.random.random()
            for _ in range(
                self.dimension
            )
        ]

        if route is None:
            return position

        if len(route) < 2:
            return position

        decision_index = 0

        for source, target in zip(
            route[:-1],
            route[1:],
        ):
            if decision_index >= self.dimension:
                break

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(
                    source
                )
                if not edge.blocked
            ]

            if not neighbors:
                continue

            target_index = None

            for index, edge in enumerate(
                neighbors
            ):
                if edge.target.id == target.id:
                    target_index = index
                    break

            if target_index is None:
                continue

            # Put the value in the middle of
            # the selected edge's interval.
            position[
                decision_index
            ] = (
                target_index + 0.5
            ) / len(neighbors)

            decision_index += 1

        return position

    # =========================================================
    # RANDOM FEASIBLE ROUTE
    # =========================================================

    def _random_feasible_route(
        self,
        start,
        destination,
    ):
        """
        Generate a random feasible route using
        randomized depth-first search.

        This is used only in research mode.

        Note:
            This recursive method should not be used
            for large-scale production/scalability tests.
        """

        start_node = self.graph.get_node(
            start
        )

        destination_node = self.graph.get_node(
            destination
        )

        if start_node.id == destination_node.id:
            return [start_node]

        def search(
            current,
            visited,
            path,
        ):
            if current.id == destination_node.id:
                return list(path)

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(
                    current
                )
                if (
                    not edge.blocked
                    and edge.target.id not in visited
                )
            ]

            self.random.shuffle(
                neighbors
            )

            for edge in neighbors:
                next_node = edge.target

                visited.add(
                    next_node.id
                )

                path.append(
                    next_node
                )

                result = search(
                    next_node,
                    visited,
                    path,
                )

                if result is not None:
                    return result

                path.pop()

                visited.remove(
                    next_node.id
                )

            return None

        return search(
            start_node,
            {start_node.id},
            [start_node],
        )

    # =========================================================
    # FIND EDGE
    # =========================================================

    def _find_edge(
        self,
        source_node,
        target_node,
    ):
        """
        Find an active edge connecting
        two nodes.
        """

        for edge in self.graph.get_neighbors(
            source_node
        ):
            if (
                edge.target.id == target_node.id
                and not edge.blocked
            ):
                return edge

        return None

    # =========================================================
    # ROUTE METRICS
    # =========================================================

    def _route_metrics(self, route):
        """
        Calculate:

        - total travel time
        - total distance
        - average congestion
        """

        if route is None or len(route) < 2:
            return (
                float("inf"),
                float("inf"),
                1.0,
            )

        total_time = 0.0
        total_distance = 0.0

        congestion_values = []

        for source_node, target_node in zip(
            route[:-1],
            route[1:],
        ):
            edge = self._find_edge(
                source_node,
                target_node,
            )

            if edge is None:
                return (
                    float("inf"),
                    float("inf"),
                    1.0,
                )

            total_time += edge.travel_time
            total_distance += edge.distance

            congestion_values.append(
                edge.congestion
            )

        if congestion_values:
            average_congestion = (
                sum(congestion_values)
                / len(congestion_values)
            )
        else:
            average_congestion = 0.0

        return (
            total_time,
            total_distance,
            average_congestion,
        )

    # =========================================================
    # ROUTE FITNESS
    # =========================================================

    def _route_cost(
        self,
        route,
        start,
        destination,
    ):
        """
        Calculate multi-objective route fitness.

        Invalid routes receive infinite cost.
        """

        if not RouteUtils.route_is_valid(
            self.graph,
            route,
            start,
            destination,
        ):
            return float("inf")

        (
            total_time,
            total_distance,
            average_congestion,
        ) = self._route_metrics(
            route
        )

        return self.fitness_function.calculate(
            travel_time=total_time,
            distance=total_distance,
            congestion=average_congestion,
        )

    # =========================================================
    # DIJKSTRA SEED
    # =========================================================

    def _get_dijkstra_seed(
        self,
        start,
        destination,
    ):
        """
        Generate a reliable feasible seed
        using Dijkstra.
        """

        try:
            dijkstra = Dijkstra(
                self.graph
            )

            result = dijkstra.optimize(
                start,
                destination,
            )

            if result is not None:
                return result.get(
                    "route"
                )

        except (
            KeyError,
            ValueError,
        ):
            pass

        return None

    # =========================================================
    # INITIALIZE PARTICLES
    # =========================================================

    def _initialize_particles(
        self,
        start,
        destination,
    ):
        """
        Create and evaluate the initial PSO population.

        Production mode:
            use_dijkstra_seed=True

            Dijkstra provides the initial feasible route.
            Random continuous positions are attempted for
            diversity, but invalid positions fall back to
            the Dijkstra route.

        Research mode:
            use_dijkstra_seed=False

            Random feasible routes are generated.
        """

        particles = []

        global_best_position = None
        global_best_route = None
        global_best_cost = float("inf")

        # -----------------------------------------------------
        # Optional Dijkstra seed
        # -----------------------------------------------------

        seed_route = None

        if self.use_dijkstra_seed:
            seed_route = self._get_dijkstra_seed(
                start,
                destination,
            )

            # Never fall back to recursive DFS in
            # production/scalability mode.
            if seed_route is None:
                raise ValueError(
                    "Dijkstra could not find a feasible route "
                    f"from {start} to {destination}."
                )

        # -----------------------------------------------------
        # Generate population
        # -----------------------------------------------------

        for particle_index in range(
            self.particle_count
        ):
            route = None

            # -------------------------------------------------
            # Production / scalability mode
            # -------------------------------------------------

            if self.use_dijkstra_seed:

                # First particle gets the reliable Dijkstra route.
                if particle_index == 0:
                    route = seed_route

                else:
                    # Generate a random position.
                    position = self._random_position()

                    # Decode it into a graph route.
                    candidate_route = (
                        self._position_to_route(
                            position,
                            start,
                            destination,
                        )
                    )

                    if candidate_route is not None:
                        route = candidate_route
                    else:
                        # Important:
                        # Do NOT use recursive DFS here.
                        route = seed_route

            # -------------------------------------------------
            # Research mode
            # -------------------------------------------------

            else:

                for _ in range(
                    self.initialization_attempts
                ):
                    candidate = (
                        self._random_feasible_route(
                            start,
                            destination,
                        )
                    )

                    if candidate is not None:
                        route = candidate
                        break

            # -------------------------------------------------
            # Ensure a feasible route exists
            # -------------------------------------------------

            if route is None:
                raise ValueError(
                    "Could not generate a feasible initial "
                    f"route from {start} to {destination}."
                )

            # -------------------------------------------------
            # Convert route to continuous position
            # -------------------------------------------------

            position = self._route_to_position(
                route
            )

            velocity = self._random_velocity()

            # -------------------------------------------------
            # Evaluate particle
            # -------------------------------------------------

            cost = self._route_cost(
                route,
                start,
                destination,
            )

            particle = {
                "position": list(position),
                "velocity": list(velocity),
                "cost": cost,
                "best_position": list(position),
                "best_route": route,
                "best_cost": cost,
            }

            particles.append(
                particle
            )

            # -------------------------------------------------
            # Update global best
            # -------------------------------------------------

            if cost < global_best_cost:
                global_best_cost = cost

                global_best_position = list(
                    position
                )

                global_best_route = route

        return (
            particles,
            global_best_position,
            global_best_route,
            global_best_cost,
        )

    # =========================================================
    # VELOCITY UPDATE
    # =========================================================

    def _update_velocity(
        self,
        velocity,
        position,
        best_position,
        global_best_position,
    ):
        """
        Classical PSO velocity update:

            v(t+1) =
                w*v(t)
                + c1*r1*(pbest-x)
                + c2*r2*(gbest-x)
        """

        new_velocity = []

        for j in range(
            self.dimension
        ):
            r1 = self.random.random()
            r2 = self.random.random()

            value = (
                self.inertia
                * velocity[j]
            )

            value += (
                self.cognitive
                * r1
                * (
                    best_position[j]
                    - position[j]
                )
            )

            value += (
                self.social
                * r2
                * (
                    global_best_position[j]
                    - position[j]
                )
            )

            # Keep velocity bounded.
            value = max(
                -1.0,
                min(
                    1.0,
                    value,
                ),
            )

            new_velocity.append(
                value
            )

        return new_velocity

    # =========================================================
    # POSITION UPDATE
    # =========================================================

    def _update_position(
        self,
        position,
        velocity,
    ):
        """
        Update particle position:

            x(t+1) = x(t) + v(t+1)
        """

        new_position = []

        for j in range(
            self.dimension
        ):
            value = (
                position[j]
                + velocity[j]
            )

            value = max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )

            new_position.append(
                value
            )

        return new_position

    # =========================================================
    # OPTIMIZE
    # =========================================================

    def optimize(
        self,
        start,
        destination,
    ):
        """
        Run classical PSO.
        """

        (
            particles,
            global_best_position,
            global_best_route,
            global_best_cost,
        ) = self._initialize_particles(
            start,
            destination,
        )

        history = []

        # -----------------------------------------------------
        # MAIN PSO LOOP
        # -----------------------------------------------------

        for iteration in range(
            self.iterations
        ):

            for particle in particles:

                new_velocity = (
                    self._update_velocity(
                        particle["velocity"],
                        particle["position"],
                        particle["best_position"],
                        global_best_position,
                    )
                )

                new_position = (
                    self._update_position(
                        particle["position"],
                        new_velocity,
                    )
                )

                new_route = (
                    self._position_to_route(
                        new_position,
                        start,
                        destination,
                    )
                )

                new_cost = (
                    self._route_cost(
                        new_route,
                        start,
                        destination,
                    )
                )

                particle["velocity"] = list(
                    new_velocity
                )

                particle["position"] = list(
                    new_position
                )

                particle["cost"] = new_cost

                # -------------------------------------------------
                # Personal best
                # -------------------------------------------------

                if (
                    new_cost
                    < particle["best_cost"]
                ):
                    particle["best_cost"] = (
                        new_cost
                    )

                    particle["best_position"] = (
                        list(new_position)
                    )

                    particle["best_route"] = (
                        new_route
                    )

                # -------------------------------------------------
                # Global best
                # -------------------------------------------------

                if (
                    new_cost
                    < global_best_cost
                ):
                    global_best_cost = (
                        new_cost
                    )

                    global_best_position = (
                        list(new_position)
                    )

                    global_best_route = (
                        new_route
                    )

            history.append(
                global_best_cost
            )

        # -----------------------------------------------------
        # FINAL METRICS
        # -----------------------------------------------------

        (
            total_time,
            total_distance,
            average_congestion,
        ) = self._route_metrics(
            global_best_route
        )

        return {
            "route": global_best_route,
            "fitness": global_best_cost,
            "travel_time": total_time,
            "distance": total_distance,
            "congestion": average_congestion,
            "position": global_best_position,
            "history": history,
            "iterations": self.iterations,
            "particles": self.particle_count,
            "inertia": self.inertia,
            "cognitive": self.cognitive,
            "social": self.social,
            "use_dijkstra_seed": self.use_dijkstra_seed,
        }

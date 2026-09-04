import math
import random

from src.algorithms.route_utils import RouteUtils
from src.optimization.fitness import FitnessFunction


class AdaptiveQPSO:
    """
    Adaptive Quantum-behaved Particle Swarm Optimization.

    Beta changes during optimization.

    beta_max:
        exploration-oriented beginning

    beta_min:
        exploitation-oriented ending

    Linear schedule:

        beta(t) =
            beta_max
            - (beta_max - beta_min) * (t / T)

    This implementation is intentionally independent of
    the project's Dijkstra implementation.
    """

    def __init__(
        self,
        graph,
        particles=20,
        iterations=50,
        beta_max=0.9,
        beta_min=0.5,
        seed=None,
        fitness_function=None,
        use_dijkstra_seed=False,
        initialization_attempts=100,
    ):
        if particles < 2:
            raise ValueError(
                "particles must be at least 2."
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be at least 1."
            )

        if beta_max <= 0:
            raise ValueError(
                "beta_max must be positive."
            )

        if beta_min <= 0:
            raise ValueError(
                "beta_min must be positive."
            )

        if beta_min > beta_max:
            raise ValueError(
                "beta_min cannot be greater than beta_max."
            )

        if initialization_attempts < 1:
            raise ValueError(
                "initialization_attempts must be at least 1."
            )

        self.graph = graph

        self.particles = int(particles)

        self.iterations = int(iterations)

        self.beta_max = float(beta_max)

        self.beta_min = float(beta_min)

        self.random = random.Random(seed)

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

        self.dimension = max(
            1,
            graph.number_of_nodes() - 1,
        )

    # ========================================================
    # ADAPTIVE BETA
    # ========================================================

    def _get_beta(self, iteration):
        """
        Calculate beta for a particular iteration.

        Example with:

            beta_max = 0.9
            beta_min = 0.5
            iterations = 50

        gives approximately:

            iteration 0   -> 0.90
            iteration 10  -> 0.82
            iteration 20  -> 0.74
            iteration 30  -> 0.66
            iteration 40  -> 0.58
            iteration 50  -> 0.50
        """

        if self.iterations == 0:
            return self.beta_min

        progress = (
            iteration / self.iterations
        )

        progress = max(
            0.0,
            min(1.0, progress),
        )

        beta = (
            self.beta_max
            - (
                self.beta_max
                - self.beta_min
            )
            * progress
        )

        return beta

    # ========================================================
    # RANDOM POSITION
    # ========================================================

    def _random_position(self):
        return [
            self.random.random()
            for _ in range(self.dimension)
        ]

    # ========================================================
    # POSITION -> ROUTE
    # ========================================================

    def _position_to_route(
        self,
        position,
        start,
        destination,
    ):
        return RouteUtils.build_route(
            self.graph,
            start,
            destination,
            position,
        )

    # ========================================================
    # ROUTE -> POSITION
    # ========================================================

    def _route_to_position(self, route):
        """
        Convert a valid route into continuous QPSO
        decision variables in [0, 1].
        """

        position = [
            self.random.random()
            for _ in range(self.dimension)
        ]

        if route is None:
            return position

        if len(route) < 2:
            return position

        for decision_index, (
            source,
            target,
        ) in enumerate(
            zip(
                route[:-1],
                route[1:],
            )
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

            selected_index = None

            for index, edge in enumerate(
                neighbors
            ):
                if edge.target.id == target.id:
                    selected_index = index
                    break

            if selected_index is None:
                continue

            position[decision_index] = (
                selected_index + 0.5
            ) / len(neighbors)

        return position

    # ========================================================
    # RANDOM FEASIBLE ROUTE
    # ========================================================

    def _random_feasible_route(
        self,
        start,
        destination,
    ):
        """
        Generate a random feasible route.

        This is used for the controlled research benchmark.

        The search is randomized so different seeds generate
        different initial populations.
        """

        start_node = self.graph.get_node(start)

        destination_node = (
            self.graph.get_node(destination)
        )

        visited = set()

        def dfs(current):
            if current.id == destination_node.id:
                return [current]

            visited.add(current.id)

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(
                    current
                )
                if (
                    not edge.blocked
                    and edge.target.id
                    not in visited
                )
            ]

            self.random.shuffle(neighbors)

            for edge in neighbors:

                result = dfs(
                    edge.target
                )

                if result is not None:
                    return [
                        current
                    ] + result

            return None

        return dfs(start_node)

    # ========================================================
    # FIND EDGE
    # ========================================================

    def _find_edge(
        self,
        source,
        target,
    ):
        for edge in self.graph.get_neighbors(
            source
        ):
            if (
                edge.target.id == target.id
                and not edge.blocked
            ):
                return edge

        return None

    # ========================================================
    # ROUTE METRICS
    # ========================================================

    def _route_metrics(self, route):
        if route is None:
            return None

        if len(route) < 2:
            return None

        total_time = 0.0

        total_distance = 0.0

        congestion_values = []

        for source, target in zip(
            route[:-1],
            route[1:],
        ):
            edge = self._find_edge(
                source,
                target,
            )

            if edge is None:
                return None

            total_time += float(
                edge.travel_time
            )

            total_distance += float(
                edge.distance
            )

            congestion_values.append(
                float(edge.congestion)
            )

        average_congestion = (
            sum(congestion_values)
            / len(congestion_values)
            if congestion_values
            else 0.0
        )

        return {
            "travel_time": total_time,
            "distance": total_distance,
            "congestion": average_congestion,
        }

    # ========================================================
    # FITNESS
    # ========================================================

    def _route_cost(self, route):
        metrics = self._route_metrics(route)

        if metrics is None:
            return float("inf")

        return self.fitness_function.calculate(
            travel_time=metrics[
                "travel_time"
            ],
            distance=metrics[
                "distance"
            ],
            congestion=metrics[
                "congestion"
            ],
        )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_particles(
        self,
        start,
        destination,
    ):
        """
        Create the initial population.

        For the current controlled benchmark we use
        randomized feasible routes.

        No Dijkstra dependency is required.
        """

        particles = []

        for _ in range(
            self.particles
        ):
            route = None

            # -----------------------------------------------
            # Find a feasible route
            # -----------------------------------------------

            for _ in range(
                self.initialization_attempts
            ):

                candidate_route = (
                    self._random_feasible_route(
                        start,
                        destination,
                    )
                )

                if candidate_route is None:
                    continue

                if RouteUtils.route_is_valid(
                    self.graph,
                    candidate_route,
                    start,
                    destination,
                ):
                    route = candidate_route
                    break

            if route is None:
                raise ValueError(
                    "Could not generate a feasible "
                    "initial route."
                )

            # -----------------------------------------------
            # Encode route
            # -----------------------------------------------

            position = (
                self._route_to_position(
                    route
                )
            )

            # -----------------------------------------------
            # Decode encoded position
            # -----------------------------------------------

            decoded_route = (
                self._position_to_route(
                    position,
                    start,
                    destination,
                )
            )

            if RouteUtils.route_is_valid(
                self.graph,
                decoded_route,
                start,
                destination,
            ):
                route = decoded_route

            else:
                # Keep the original feasible route.
                position = (
                    self._route_to_position(
                        route
                    )
                )

            cost = self._route_cost(
                route
            )

            particles.append(
                {
                    "position": position,

                    "cost": cost,

                    "route": route,

                    "best_position": (
                        position.copy()
                    ),

                    "best_cost": cost,

                    "best_route": route,
                }
            )

        return particles

    # ========================================================
    # MEAN BEST
    # ========================================================

    def _mean_best(self, particles):
        mean_best = []

        for dimension_index in range(
            self.dimension
        ):

            value = sum(
                particle[
                    "best_position"
                ][dimension_index]
                for particle in particles
            ) / len(particles)

            mean_best.append(value)

        return mean_best

    # ========================================================
    # QPSO UPDATE
    # ========================================================

    def _update_position(
        self,
        position,
        best_position,
        global_best_position,
        mean_best,
        beta,
    ):
        """
        Quantum-inspired particle update.
        """

        new_position = []

        for j in range(
            self.dimension
        ):

            # Avoid log(0)
            u = max(
                self.random.random(),
                1e-12,
            )

            phi = self.random.random()

            # -----------------------------------------------
            # Local attractor
            # -----------------------------------------------

            local_attractor = (
                phi * best_position[j]
                + (
                    1.0 - phi
                )
                * global_best_position[j]
            )

            # -----------------------------------------------
            # Quantum step
            # -----------------------------------------------

            step = (
                beta
                * abs(
                    mean_best[j]
                    - position[j]
                )
                * math.log(
                    1.0 / u
                )
            )

            # -----------------------------------------------
            # Random direction
            # -----------------------------------------------

            if self.random.random() < 0.5:
                new_value = (
                    local_attractor
                    + step
                )
            else:
                new_value = (
                    local_attractor
                    - step
                )

            # -----------------------------------------------
            # Clamp
            # -----------------------------------------------

            new_value = max(
                0.0,
                min(
                    1.0,
                    new_value,
                ),
            )

            new_position.append(
                new_value
            )

        return new_position

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    def optimize(
        self,
        start,
        destination,
    ):
        """
        Run Adaptive-QPSO.

        Returns:
            route
            fitness
            travel_time
            distance
            congestion
            position
            history
            beta_history
        """

        particles = (
            self._initialize_particles(
                start,
                destination,
            )
        )

        # ====================================================
        # INITIAL GLOBAL BEST
        # ====================================================

        global_best_particle = min(
            particles,
            key=lambda particle:
                particle["best_cost"],
        )

        global_best_cost = (
            global_best_particle[
                "best_cost"
            ]
        )

        global_best_position = (
            global_best_particle[
                "best_position"
            ].copy()
        )

        global_best_route = (
            global_best_particle[
                "best_route"
            ]
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history = [
            global_best_cost
        ]

        beta_history = [
            self._get_beta(0)
        ]

        # ====================================================
        # MAIN OPTIMIZATION LOOP
        # ====================================================

        for iteration in range(
            1,
            self.iterations + 1,
        ):

            # -----------------------------------------------
            # Adaptive beta
            # -----------------------------------------------

            beta = self._get_beta(
                iteration
            )

            beta_history.append(
                beta
            )

            # -----------------------------------------------
            # Mean personal-best position
            # -----------------------------------------------

            mean_best = (
                self._mean_best(
                    particles
                )
            )

            # -----------------------------------------------
            # Update particles
            # -----------------------------------------------

            for particle in particles:

                new_position = (
                    self._update_position(
                        position=particle[
                            "position"
                        ],
                        best_position=particle[
                            "best_position"
                        ],
                        global_best_position=(
                            global_best_position
                        ),
                        mean_best=mean_best,
                        beta=beta,
                    )
                )

                new_route = (
                    self._position_to_route(
                        new_position,
                        start,
                        destination,
                    )
                )

                # -------------------------------------------
                # Invalid route
                # -------------------------------------------

                if not RouteUtils.route_is_valid(
                    self.graph,
                    new_route,
                    start,
                    destination,
                ):
                    continue

                # -------------------------------------------
                # Fitness
                # -------------------------------------------

                new_cost = (
                    self._route_cost(
                        new_route
                    )
                )

                particle["position"] = (
                    new_position
                )

                particle["route"] = (
                    new_route
                )

                particle["cost"] = (
                    new_cost
                )

                # ===========================================
                # PERSONAL BEST
                # ===========================================

                if (
                    new_cost
                    < particle["best_cost"]
                ):

                    particle[
                        "best_cost"
                    ] = new_cost

                    particle[
                        "best_position"
                    ] = (
                        new_position.copy()
                    )

                    particle[
                        "best_route"
                    ] = new_route

                # ===========================================
                # GLOBAL BEST
                # ===========================================

                if (
                    particle["best_cost"]
                    < global_best_cost
                ):

                    global_best_cost = (
                        particle[
                            "best_cost"
                        ]
                    )

                    global_best_position = (
                        particle[
                            "best_position"
                        ].copy()
                    )

                    global_best_route = (
                        particle[
                            "best_route"
                        ]
                    )

            # -----------------------------------------------
            # Record convergence
            # -----------------------------------------------

            history.append(
                global_best_cost
            )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        metrics = self._route_metrics(
            global_best_route
        )

        if metrics is None:
            raise ValueError(
                "Adaptive QPSO did not produce "
                "a valid final route."
            )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "route": global_best_route,

            "fitness": global_best_cost,

            "travel_time": metrics[
                "travel_time"
            ],

            "distance": metrics[
                "distance"
            ],

            "congestion": metrics[
                "congestion"
            ],

            "position": (
                global_best_position
            ),

            "history": history,

            "beta_history": beta_history,

            "iterations": self.iterations,

            "particles": self.particles,

            "beta_max": self.beta_max,

            "beta_min": self.beta_min,

            "use_dijkstra_seed": False,
        }

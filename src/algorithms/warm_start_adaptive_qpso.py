import math
import random

from src.algorithms.route_utils import RouteUtils
from src.optimization.fitness import FitnessFunction


class WarmStartAdaptiveQPSO:
    """
    Adaptive Quantum Particle Swarm Optimization with warm-start support.

    Features:
    - Quantum-inspired particle update
    - Adaptive beta from beta_max to beta_min
    - Optional warm-start using a previous valid route
    - Feasible-route initialization
    - Traffic-aware fitness evaluation
    - No dependency on Dijkstra
    - Route IDs / Node objects are handled consistently
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
        initial_route=None,
        initialization_attempts=100,
    ):
        if particles < 2:
            raise ValueError("particles must be at least 2.")

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")

        if beta_max < 0:
            raise ValueError("beta_max must be non-negative.")

        if beta_min < 0:
            raise ValueError("beta_min must be non-negative.")

        if beta_min > beta_max:
            raise ValueError("beta_min cannot be greater than beta_max.")

        if initialization_attempts < 1:
            raise ValueError(
                "initialization_attempts must be at least 1."
            )

        self.graph = graph
        self.particles = int(particles)
        self.iterations = int(iterations)

        self.beta_max = float(beta_max)
        self.beta_min = float(beta_min)

        self.seed = seed
        self.random = random.Random(seed)

        self.fitness_function = (
            fitness_function
            if fitness_function is not None
            else FitnessFunction()
        )

        self.initial_route = initial_route
        self.initialization_attempts = int(initialization_attempts)

        # One decision variable per possible hop.
        self.dimension = max(
            1,
            self.graph.number_of_nodes() - 1,
        )

    # ============================================================
    # Helper: Node ID conversion
    # ============================================================

    @staticmethod
    def _node_id(node):
        """
        Return a node ID regardless of whether the caller provides
        a Node object or a raw node ID.
        """
        if hasattr(node, "id"):
            return str(node.id)

        return str(node)

    # ============================================================
    # Adaptive beta
    # ============================================================

    def _get_beta(self, iteration):
        """
        Linearly decrease beta from beta_max to beta_min.

        iteration:
            0 ... iterations-1
        """

        if self.iterations <= 1:
            return self.beta_min

        progress = iteration / (self.iterations - 1)

        beta = (
            self.beta_max
            - progress * (self.beta_max - self.beta_min)
        )

        return beta

    # ============================================================
    # Random position
    # ============================================================

    def _random_position(self):
        """
        Generate a continuous QPSO position vector in [0, 1].
        """

        return [
            self.random.random()
            for _ in range(self.dimension)
        ]

    # ============================================================
    # Position -> Route
    # ============================================================

    def _position_to_route(
        self,
        start,
        destination,
        position,
    ):
        """
        Decode a continuous QPSO position into a graph route.

        This method explicitly converts Node objects to node IDs
        before calling RouteUtils.build_route().
        """

        start_id = self._node_id(start)
        destination_id = self._node_id(destination)

        return RouteUtils.build_route(
            self.graph,
            start_id,
            destination_id,
            position,
        )

    # ============================================================
    # Find edge
    # ============================================================

    def _find_edge(self, source, target):
        """
        Find the non-blocked edge from source to target.

        Returns:
            Edge object or None
        """

        source_id = self._node_id(source)
        target_id = self._node_id(target)

        source_node = self.graph.get_node(source_id)

        for edge in self.graph.get_neighbors(source_node):
            if edge.blocked:
                continue

            if edge.target.id == target_id:
                return edge

        return None

    # ============================================================
    # Route -> Position
    # ============================================================

    def _route_to_position(self, route):
        """
        Encode a valid route into a continuous QPSO position vector.

        Each selected outgoing edge is represented by the center
        of its corresponding decision interval.
        """

        position = [
            0.5
            for _ in range(self.dimension)
        ]

        if route is None or len(route) < 2:
            return position

        for i in range(len(route) - 1):

            if i >= self.dimension:
                break

            source = route[i]
            target = route[i + 1]

            source_id = self._node_id(source)
            target_id = self._node_id(target)

            source_node = self.graph.get_node(source_id)

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(source_node)
                if not edge.blocked
            ]

            if not neighbors:
                continue

            selected_index = None

            for edge_index, edge in enumerate(neighbors):

                edge_target_id = self._node_id(
                    edge.target
                )

                if edge_target_id == target_id:
                    selected_index = edge_index
                    break

            if selected_index is not None:

                position[i] = (
                    selected_index + 0.5
                ) / len(neighbors)

        return position

    # ============================================================
    # Random feasible route
    # ============================================================

    def _random_feasible_route(
        self,
        start,
        destination,
    ):
        """
        Generate a random feasible route.

        This is intended for small/controlled benchmark graphs.

        It uses randomized depth-first search with a visited set
        so cycles are avoided.
        """

        start_id = self._node_id(start)
        destination_id = self._node_id(destination)

        start_node = self.graph.get_node(start_id)
        destination_node = self.graph.get_node(destination_id)

        def dfs(current, visited, route):

            if current.id == destination_node.id:
                return route.copy()

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(current)
                if not edge.blocked
                and edge.target.id not in visited
            ]

            self.random.shuffle(neighbors)

            for edge in neighbors:

                visited.add(edge.target.id)
                route.append(edge.target)

                result = dfs(
                    edge.target,
                    visited,
                    route,
                )

                if result is not None:
                    return result

                route.pop()
                visited.remove(edge.target.id)

            return None

        return dfs(
            start_node,
            {start_node.id},
            [start_node],
        )

    # ============================================================
    # Route metrics
    # ============================================================

    def _route_metrics(self, route):
        """
        Calculate total travel time, distance and average congestion.
        """

        if route is None or len(route) < 2:
            return None

        total_time = 0.0
        total_distance = 0.0
        total_congestion = 0.0
        edge_count = 0

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

            if edge.blocked:
                return None

            total_time += float(edge.travel_time)
            total_distance += float(edge.distance)
            total_congestion += float(edge.congestion)

            edge_count += 1

        if edge_count == 0:
            return None

        average_congestion = (
            total_congestion / edge_count
        )

        return (
            total_time,
            total_distance,
            average_congestion,
        )

    # ============================================================
    # Route cost
    # ============================================================

    def _route_cost(self, route):
        """
        Calculate fitness of a route.

        Invalid routes receive infinity.
        """

        metrics = self._route_metrics(route)

        if metrics is None:
            return math.inf

        travel_time, distance, congestion = metrics

        return self.fitness_function.calculate(
            travel_time=travel_time,
            distance=distance,
            congestion=congestion,
        )

    # ============================================================
    # Particle initialization
    # ============================================================

    def _initialize_particles(
        self,
        start_node,
        destination_node,
    ):
        """
        Initialize particles.

        Particle 0:
            - Uses initial_route when a valid warm-start route
              is supplied.

        Remaining particles:
            - Use randomly generated feasible routes.

        If no warm-start route exists:
            - Particle 0 is also randomly initialized.
        """

        start_id = self._node_id(start_node)
        destination_id = self._node_id(destination_node)

        start = self.graph.get_node(start_id)
        destination = self.graph.get_node(destination_id)

        particles = []

        # --------------------------------------------------------
        # Validate warm-start route
        # --------------------------------------------------------

        warm_route = None

        if self.initial_route is not None:

            try:

                if RouteUtils.route_is_valid(
                    self.graph,
                    self.initial_route,
                    start_id,
                    destination_id,
                ):
                    warm_route = list(
                        self.initial_route
                    )

            except Exception:
                warm_route = None

        # --------------------------------------------------------
        # Create particles
        # --------------------------------------------------------

        for particle_index in range(
            self.particles
        ):

            route = None

            # ----------------------------------------------------
            # Particle 0 = warm-start particle
            # ----------------------------------------------------

            if (
                particle_index == 0
                and warm_route is not None
            ):

                route = list(warm_route)

            # ----------------------------------------------------
            # Other particles = random feasible routes
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Initialization failure
            # ----------------------------------------------------

            if route is None:

                raise RuntimeError(
                    "Unable to generate a feasible "
                    "initial route from "
                    f"{start_id} to {destination_id}."
                )

            # ----------------------------------------------------
            # Verify route
            # ----------------------------------------------------

            if not RouteUtils.route_is_valid(
                self.graph,
                route,
                start_id,
                destination_id,
            ):
                raise RuntimeError(
                    "Generated initialization route "
                    "is invalid."
                )

            # ----------------------------------------------------
            # Encode route
            # ----------------------------------------------------

            position = self._route_to_position(
                route
            )

            # ----------------------------------------------------
            # Decode again for consistency
            # ----------------------------------------------------

            decoded_route = (
                self._position_to_route(
                    start,
                    destination,
                    position,
                )
            )

            if (
                decoded_route is None
                or not RouteUtils.route_is_valid(
                    self.graph,
                    decoded_route,
                    start_id,
                    destination_id,
                )
            ):
                # Keep the original known-valid route.
                decoded_route = list(route)

            # ----------------------------------------------------
            # Calculate cost
            # ----------------------------------------------------

            cost = self._route_cost(
                decoded_route
            )

            if not math.isfinite(cost):
                raise RuntimeError(
                    "Generated initialization route "
                    "has invalid fitness."
                )

            particle = {
                "position": list(position),
                "cost": cost,
                "best_position": list(position),
                "best_cost": cost,
                "best_route": list(decoded_route),
            }

            particles.append(particle)

        return particles, warm_route

    # ============================================================
    # Mean best position
    # ============================================================

    def _mean_best(self, particles):
        """
        Calculate mean of all particles' personal-best positions.
        """

        mean_position = [
            0.0
            for _ in range(self.dimension)
        ]

        for particle in particles:

            best_position = particle[
                "best_position"
            ]

            for j in range(self.dimension):

                mean_position[j] += (
                    best_position[j]
                )

        for j in range(self.dimension):

            mean_position[j] /= len(particles)

        return mean_position

    # ============================================================
    # QPSO position update
    # ============================================================

    def _update_position(
        self,
        position,
        best_position,
        global_best_position,
        mean_best,
        beta,
    ):
        """
        Quantum-inspired particle position update.

        Formula:

            local_attractor =
                phi * personal_best
                +
                (1 - phi) * global_best

            step =
                beta *
                |mean_best - position| *
                ln(1/u)

            new_position =
                local_attractor +/- step

        The position is clamped to [0, 1].
        """

        new_position = [
            0.0
            for _ in range(self.dimension)
        ]

        for j in range(self.dimension):

            u = max(
                self.random.random(),
                1e-12,
            )

            phi = self.random.random()

            local_attractor = (
                phi * best_position[j]
                + (1.0 - phi)
                * global_best_position[j]
            )

            step = (
                beta
                * abs(
                    mean_best[j]
                    - position[j]
                )
                * math.log(1.0 / u)
            )

            if self.random.random() < 0.5:
                new_value = (
                    local_attractor + step
                )
            else:
                new_value = (
                    local_attractor - step
                )

            # Clamp to valid continuous range.
            new_value = max(
                0.0,
                min(1.0, new_value),
            )

            new_position[j] = new_value

        return new_position

    # ============================================================
    # Optimization
    # ============================================================

    def optimize(
        self,
        start_node,
        destination_node,
    ):
        """
        Run Adaptive-QPSO.

        Returns:
            dict containing:
                route
                fitness
                travel_time
                distance
                congestion
                position
                history
                beta_history
                iterations
                particles
                beta_max
                beta_min
                warm_started
        """

        start_id = self._node_id(
            start_node
        )

        destination_id = self._node_id(
            destination_node
        )

        # --------------------------------------------------------
        # Initialize particles
        # --------------------------------------------------------

        particles, warm_route = (
            self._initialize_particles(
                start_id,
                destination_id,
            )
        )

        # --------------------------------------------------------
        # Find initial global best
        # --------------------------------------------------------

        global_best_particle = min(
            particles,
            key=lambda particle: particle[
                "best_cost"
            ],
        )

        global_best_position = list(
            global_best_particle[
                "best_position"
            ]
        )

        global_best_cost = (
            global_best_particle[
                "best_cost"
            ]
        )

        global_best_route = list(
            global_best_particle[
                "best_route"
            ]
        )

        # --------------------------------------------------------
        # Initial metrics
        # --------------------------------------------------------

        initial_metrics = self._route_metrics(
            global_best_route
        )

        if initial_metrics is None:
            raise RuntimeError(
                "Initial global-best route is invalid."
            )

        # --------------------------------------------------------
        # History
        # --------------------------------------------------------

        history = [
            global_best_cost
        ]

        beta_history = []

        # --------------------------------------------------------
        # Optimization iterations
        # --------------------------------------------------------

        for iteration in range(
            self.iterations
        ):

            beta = self._get_beta(
                iteration
            )

            beta_history.append(beta)

            mean_best = self._mean_best(
                particles
            )

            # ----------------------------------------------------
            # Update every particle
            # ----------------------------------------------------

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

                route = (
                    self._position_to_route(
                        start_id,
                        destination_id,
                        new_position,
                    )
                )

                # ------------------------------------------------
                # Evaluate candidate
                # ------------------------------------------------

                if route is None:

                    cost = math.inf

                elif not RouteUtils.route_is_valid(
                    self.graph,
                    route,
                    start_id,
                    destination_id,
                ):

                    cost = math.inf

                else:

                    cost = self._route_cost(
                        route
                    )

                # ------------------------------------------------
                # Update current position
                # ------------------------------------------------

                particle[
                    "position"
                ] = list(new_position)

                # ------------------------------------------------
                # Personal best
                # ------------------------------------------------

                if cost < particle[
                    "best_cost"
                ]:

                    particle[
                        "best_cost"
                    ] = cost

                    particle[
                        "best_position"
                    ] = list(new_position)

                    particle[
                        "best_route"
                    ] = list(route)

            # ----------------------------------------------------
            # Global best
            # ----------------------------------------------------

            candidate_particle = min(
                particles,
                key=lambda particle: particle[
                    "best_cost"
                ],
            )

            candidate_cost = (
                candidate_particle[
                    "best_cost"
                ]
            )

            if candidate_cost < global_best_cost:

                global_best_cost = (
                    candidate_cost
                )

                global_best_position = list(
                    candidate_particle[
                        "best_position"
                    ]
                )

                global_best_route = list(
                    candidate_particle[
                        "best_route"
                    ]
                )

            history.append(
                global_best_cost
            )

        # --------------------------------------------------------
        # Final metrics
        # --------------------------------------------------------

        metrics = self._route_metrics(
            global_best_route
        )

        if metrics is None:
            raise RuntimeError(
                "Final global-best route is invalid."
            )

        travel_time, distance, congestion = (
            metrics
        )

        # --------------------------------------------------------
        # Final route validation
        # --------------------------------------------------------

        if not RouteUtils.route_is_valid(
            self.graph,
            global_best_route,
            start_id,
            destination_id,
        ):
            raise RuntimeError(
                "Optimization produced an invalid route."
            )

        # --------------------------------------------------------
        # Return result
        # --------------------------------------------------------

        return {
            "route": global_best_route,
            "fitness": global_best_cost,
            "travel_time": travel_time,
            "distance": distance,
            "congestion": congestion,
            "position": global_best_position,
            "history": history,
            "beta_history": beta_history,
            "iterations": self.iterations,
            "particles": self.particles,
            "beta_max": self.beta_max,
            "beta_min": self.beta_min,
            "warm_started": warm_route is not None,
        }
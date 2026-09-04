import math
import random

from src.algorithms.dijkstra import Dijkstra
from src.algorithms.route_utils import RouteUtils
from src.optimization.fitness import FitnessFunction


class QPSO:
    """
    Quantum-inspired Particle Swarm Optimization (QPSO)
    for traffic-route optimization.

    Two initialization modes are supported:

    use_dijkstra_seed=True
        Reliable Dijkstra-based initialization suitable for
        production/scalability experiments.

    use_dijkstra_seed=False
        Diverse randomized feasible-route initialization
        for controlled research experiments.

    Lower fitness is better.
    """

    def __init__(
        self,
        graph,
        particles=20,
        iterations=50,
        beta=0.7,
        seed=None,
        fitness_function=None,
        use_dijkstra_seed=True,
        initialization_attempts=100,
    ):
        if particles < 2:
            raise ValueError("particles must be at least 2.")

        if iterations < 1:
            raise ValueError("iterations must be positive.")

        if beta <= 0:
            raise ValueError("beta must be positive.")

        if initialization_attempts < 1:
            raise ValueError(
                "initialization_attempts must be positive."
            )

        self.graph = graph

        self.particle_count = int(particles)
        self.iterations = int(iterations)
        self.beta = float(beta)

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

        self.edges = graph.get_edges()

        if not self.edges:
            raise ValueError(
                "Graph must contain at least one edge."
            )

        self.dimension = max(
            1,
            graph.number_of_nodes() - 1,
        )

    # =========================================================
    # RANDOM POSITION
    # =========================================================

    def _random_position(self):
        """
        Generate a random QPSO position.
        """

        return [
            self.random.random()
            for _ in range(self.dimension)
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
        Convert continuous QPSO position into
        a discrete graph route.
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
        Encode a valid route into QPSO's
        continuous position representation.
        """

        position = [
            self.random.random()
            for _ in range(self.dimension)
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
                for edge in self.graph.get_neighbors(source)
                if not edge.blocked
            ]

            if not neighbors:
                continue

            target_index = None

            for index, edge in enumerate(neighbors):
                if edge.target.id == target.id:
                    target_index = index
                    break

            if target_index is None:
                continue

            position[decision_index] = (
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
        Generate a randomized feasible route using
        randomized depth-first search.

        IMPORTANT:
        This method is only intended for controlled
        research experiments.

        Do NOT use this method for large-scale graphs.
        Production/scalability mode uses Dijkstra instead.
        """

        start_node = self.graph.get_node(start)
        destination_node = self.graph.get_node(
            destination
        )

        if start_node.id == destination_node.id:
            return [start_node]

        def search(current, visited, path):
            if current.id == destination_node.id:
                return list(path)

            neighbors = [
                edge
                for edge in self.graph.get_neighbors(current)
                if not edge.blocked
                and edge.target.id not in visited
            ]

            self.random.shuffle(neighbors)

            for edge in neighbors:
                next_node = edge.target

                visited.add(next_node.id)
                path.append(next_node)

                result = search(
                    next_node,
                    visited,
                    path,
                )

                if result is not None:
                    return result

                path.pop()
                visited.remove(next_node.id)

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
        Find active edge between two nodes.
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
        Calculate route metrics.

        Returns:
            travel_time
            distance
            average_congestion
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
        Calculate route fitness.
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
        ) = self._route_metrics(route)

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
        Obtain a feasible Dijkstra route.

        Dijkstra is used only to provide a reliable
        initial feasible solution. QPSO still performs
        the subsequent population updates.
        """

        try:
            dijkstra = Dijkstra(self.graph)

            result = dijkstra.optimize(
                start,
                destination,
            )

            if result is not None:
                return result.get("route")

        except (
            KeyError,
            ValueError,
        ):
            pass

        return None

    # =========================================================
    # MEAN BEST
    # =========================================================

    def _mean_best(self, particles):
        """
        Calculate mean best position:

            mbest = mean(pbest_i)
        """

        mean_position = []

        for j in range(self.dimension):
            value = (
                sum(
                    particle["best_position"][j]
                    for particle in particles
                )
                / len(particles)
            )

            mean_position.append(value)

        return mean_position

    # =========================================================
    # INITIALIZE PARTICLES
    # =========================================================

    def _initialize_particles(
        self,
        start,
        destination,
    ):
        """
        Initialize QPSO population.

        Dijkstra mode:
            - Obtain one reliable Dijkstra route.
            - First particle uses the Dijkstra route.
            - Remaining particles start from random
              continuous positions.
            - If a random position cannot decode to a
              feasible route, directly use the Dijkstra
              route as fallback.
            - NEVER call recursive random DFS.

        Research mode:
            - Randomized feasible-route initialization
              is used.
        """

        particles = []

        global_best_position = None
        global_best_route = None
        global_best_cost = float("inf")

        seed_route = None

        # -----------------------------------------------------
        # GET DIJKSTRA SEED
        # -----------------------------------------------------

        if self.use_dijkstra_seed:
            seed_route = self._get_dijkstra_seed(
                start,
                destination,
            )

            if seed_route is None:
                raise ValueError(
                    "Dijkstra could not find a feasible "
                    f"route from {start} to {destination}."
                )

        # -----------------------------------------------------
        # CREATE PARTICLES
        # -----------------------------------------------------

        for particle_index in range(
            self.particle_count
        ):

            route = None

            # =================================================
            # PRODUCTION / SCALABILITY MODE
            # =================================================

            if self.use_dijkstra_seed:

                # First particle = reliable Dijkstra solution
                if particle_index == 0:

                    route = seed_route

                else:

                    # Generate random continuous QPSO position
                    position = self._random_position()

                    # Try decoding the random position
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

                        # IMPORTANT:
                        # Never call recursive DFS here.
                        route = seed_route

            # =================================================
            # RESEARCH MODE
            # =================================================

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
            # SAFETY FALLBACK
            # -------------------------------------------------

            if route is None:

                if self.use_dijkstra_seed:
                    route = seed_route

                else:
                    raise ValueError(
                        "QPSO could not create a feasible "
                        f"route from {start} to {destination} "
                        "during initialization."
                    )

            # -------------------------------------------------
            # ENCODE ROUTE
            # -------------------------------------------------

            position = self._route_to_position(
                route
            )

            # -------------------------------------------------
            # VERIFY DECODED ROUTE
            # -------------------------------------------------

            decoded_route = (
                self._position_to_route(
                    position,
                    start,
                    destination,
                )
            )

            # If encoding/decoding fails, preserve the
            # original known-feasible route.
            if decoded_route is None:

                decoded_route = route

                position = (
                    self._route_to_position(
                        route
                    )
                )

            # -------------------------------------------------
            # CALCULATE FITNESS
            # -------------------------------------------------

            cost = self._route_cost(
                decoded_route,
                start,
                destination,
            )

            # -------------------------------------------------
            # CREATE PARTICLE
            # -------------------------------------------------

            particle = {
                "position": list(position),
                "cost": cost,
                "best_position": list(position),
                "best_cost": cost,
                "best_route": decoded_route,
            }

            particles.append(particle)

            # -------------------------------------------------
            # GLOBAL BEST
            # -------------------------------------------------

            if cost < global_best_cost:

                global_best_cost = cost

                global_best_position = list(
                    position
                )

                global_best_route = decoded_route

        # -----------------------------------------------------
        # VERIFY INITIALIZATION
        # -----------------------------------------------------

        if global_best_position is None:

            raise ValueError(
                "QPSO could not find a feasible "
                f"route from {start} to "
                f"{destination} during initialization."
            )

        return (
            particles,
            global_best_position,
            global_best_route,
            global_best_cost,
        )

    # =========================================================
    # QPSO POSITION UPDATE
    # =========================================================

    def _update_position(
        self,
        position,
        best_position,
        mean_best,
    ):
        """
        Standard QPSO position update.

        Local attractor:

            p =
                phi * pbest
                + (1 - phi) * mbest

        Step:

            step =
                beta *
                |mbest - x| *
                ln(1/u)

        New position:

            x(t+1) = p +/- step
        """

        new_position = []

        for j in range(self.dimension):

            # Random number for logarithmic step
            u = self.random.random()

            # Prevent log(0)
            u = max(
                u,
                1e-12,
            )

            # Random interpolation factor
            phi = self.random.random()

            # ---------------------------------------------
            # LOCAL ATTRACTOR
            # ---------------------------------------------

            local_attractor = (
                phi * best_position[j]
                + (
                    1.0 - phi
                ) * mean_best[j]
            )

            # ---------------------------------------------
            # QUANTUM STEP
            # ---------------------------------------------

            step = (
                self.beta
                * abs(
                    mean_best[j]
                    - position[j]
                )
                * math.log(
                    1.0 / u
                )
            )

            # ---------------------------------------------
            # RANDOM DIRECTION
            # ---------------------------------------------

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

            # ---------------------------------------------
            # BOUND POSITION
            # ---------------------------------------------

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

    # =========================================================
    # OPTIMIZE
    # =========================================================

    def optimize(
        self,
        start,
        destination,
    ):
        """
        Run QPSO optimization.
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

        # Store convergence history.
        #
        # IMPORTANT:
        # Do not store the initial fitness here.
        # The history contains exactly one value
        # for each optimization iteration.
        history = []

        # -----------------------------------------------------
        # MAIN QPSO LOOP
        # -----------------------------------------------------

        for iteration in range(
            self.iterations
        ):

            # Calculate mean best position
            mean_best = (
                self._mean_best(
                    particles
                )
            )

            # -------------------------------------------------
            # UPDATE EACH PARTICLE
            # -------------------------------------------------

            for particle in particles:

                new_position = (
                    self._update_position(
                        particle["position"],
                        particle["best_position"],
                        mean_best,
                    )
                )

                # ---------------------------------------------
                # DECODE NEW POSITION
                # ---------------------------------------------

                new_route = (
                    self._position_to_route(
                        new_position,
                        start,
                        destination,
                    )
                )

                # ---------------------------------------------
                # CALCULATE NEW FITNESS
                # ---------------------------------------------

                new_cost = (
                    self._route_cost(
                        new_route,
                        start,
                        destination,
                    )
                )

                # ---------------------------------------------
                # UPDATE CURRENT PARTICLE
                # ---------------------------------------------

                particle["position"] = list(
                    new_position
                )

                particle["cost"] = new_cost

                # ---------------------------------------------
                # PERSONAL BEST
                # ---------------------------------------------

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

                # ---------------------------------------------
                # GLOBAL BEST
                # ---------------------------------------------

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

            # -------------------------------------------------
            # CONVERGENCE HISTORY
            # -------------------------------------------------

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

        # -----------------------------------------------------
        # RETURN RESULT
        # -----------------------------------------------------

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
            "beta": self.beta,
            "use_dijkstra_seed": self.use_dijkstra_seed,
        }
"""
Particle Swarm Optimization for the Capacitated Vehicle Routing Problem.

Representation:
    Each customer receives a continuous random key in [0, 1].
    Sorting the keys creates a customer permutation.
    The permutation is decoded into capacity-feasible vehicle routes.

Lower fitness is better.
"""

import math
import random

from src.algorithms.vrp_route_utils import (
    empty_solution,
    all_capacity_constraints_satisfied,
    clone_solution,
)

from src.optimization.vrp_fitness import (
    VRPFitnessFunction,
)


class VRPPSO:
    """
    Particle Swarm Optimization for the Capacitated Vehicle Routing Problem.
    """

    def __init__(
        self,
        problem,
        particles=20,
        iterations=50,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=None,
        fitness_function=None,
    ):
        if problem is None:
            raise ValueError("problem is required.")

        if particles < 2:
            raise ValueError(
                "particles must be at least 2."
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be at least 1."
            )

        if inertia < 0:
            raise ValueError(
                "inertia cannot be negative."
            )

        if cognitive < 0:
            raise ValueError(
                "cognitive cannot be negative."
            )

        if social < 0:
            raise ValueError(
                "social cannot be negative."
            )

        self.problem = problem

        self.particles = int(particles)
        self.iterations = int(iterations)

        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)

        self.random = random.Random(seed)

        self.fitness_function = (
            fitness_function
            if fitness_function is not None
            else VRPFitnessFunction()
        )

        self.dimension = max(
            1,
            self.problem.number_of_customers,
        )

    # ================================================================
    # RANDOM POSITION
    # ================================================================

    def _random_position(self):
        """
        Generate a random continuous position.

        Each dimension corresponds to one customer.
        """

        return [
            self.random.random()
            for _ in range(self.dimension)
        ]

    # ================================================================
    # RANDOM VELOCITY
    # ================================================================

    def _random_velocity(self):
        """
        Generate a random initial velocity.
        """

        return [
            self.random.uniform(-0.5, 0.5)
            for _ in range(self.dimension)
        ]

    # ================================================================
    # CUSTOMER ORDER
    # ================================================================

    def _decode_customer_order(self, position):
        """
        Convert a continuous position into a customer permutation.

        Example:

            position = [0.9, 0.1, 0.5]

            customer IDs = [C1, C2, C3]

            sorted keys:

                C2 -> 0.1
                C3 -> 0.5
                C1 -> 0.9

            result:

                [C2, C3, C1]
        """

        customer_ids = [
            customer.id
            for customer in self.problem.customers
        ]

        if len(position) != len(customer_ids):
            raise ValueError(
                "Position dimension does not match "
                "number of customers."
            )

        pairs = list(
            zip(
                customer_ids,
                position,
            )
        )

        pairs.sort(
            key=lambda item: item[1]
        )

        return [
            customer_id
            for customer_id, _ in pairs
        ]

    # Backward-compatible internal name.
    def _customer_order_from_position(
        self,
        position,
    ):
        return self._decode_customer_order(
            position
        )

    # ================================================================
    # CAPACITY-AWARE DECODER
    # ================================================================

    def _split_into_vehicle_routes(
        self,
        customer_order,
    ):
        """
        Convert a customer permutation into vehicle routes.

        Best-fit capacity assignment is used to reduce the number
        of infeasible particles.
        """

        solution = empty_solution(
            self.problem
        )

        remaining_capacity = {
            vehicle.id: float(
                vehicle.capacity
            )
            for vehicle in self.problem.vehicles
        }

        vehicle_ids = [
            vehicle.id
            for vehicle in self.problem.vehicles
        ]

        for customer_id in customer_order:

            if customer_id not in self.problem.customer_map:
                return None

            customer = (
                self.problem.customer_map[
                    customer_id
                ]
            )

            demand = float(
                customer.demand
            )

            feasible_vehicles = [
                vehicle_id
                for vehicle_id in vehicle_ids
                if remaining_capacity[
                    vehicle_id
                ] >= demand
            ]

            if not feasible_vehicles:
                return None

            # Best-fit assignment.
            assigned_vehicle = min(
                feasible_vehicles,
                key=lambda vehicle_id:
                    remaining_capacity[
                        vehicle_id
                    ] - demand,
            )

            solution[
                assigned_vehicle
            ].append(
                customer_id
            )

            remaining_capacity[
                assigned_vehicle
            ] -= demand

        if not all_capacity_constraints_satisfied(
            self.problem,
            solution,
        ):
            return None

        return solution

    # ================================================================
    # POSITION DECODING
    # ================================================================

    def decode_position(
        self,
        position,
    ):
        """
        Convert a particle position into a VRP solution.
        """

        if position is None:
            return None

        customer_order = (
            self._decode_customer_order(
                position
            )
        )

        return self._split_into_vehicle_routes(
            customer_order
        )

    # ================================================================
    # POSITION EVALUATION
    # ================================================================

    def evaluate_position(
        self,
        position,
    ):
        """
        Evaluate a particle position.

        Returns a dictionary containing:

            valid
            solution
            fitness
            travel_time
            distance
            congestion
            vehicle_results
        """

        solution = self.decode_position(
            position
        )

        if solution is None:

            return {
                "valid": False,
                "solution": None,
                "fitness": float("inf"),
                "travel_time": float("inf"),
                "distance": float("inf"),
                "congestion": float("inf"),
                "vehicle_results": [],
            }

        try:

            result = (
                self.fitness_function.evaluate(
                    self.problem,
                    solution,
                )
            )

            # --------------------------------------------------------
            # VRPFitnessFunction returns the fitness and vehicle
            # results. We aggregate the vehicle metrics here so that
            # the PSO API exposes convenient top-level values.
            # --------------------------------------------------------

            vehicle_results = result.get(
                "vehicle_results",
                [],
            )

            total_travel_time = 0.0
            total_distance = 0.0

            congestion_values = []

            if isinstance(
                vehicle_results,
                dict,
            ):
                vehicle_iterable = (
                    vehicle_results.values()
                )
            else:
                vehicle_iterable = (
                    vehicle_results
                )

            for vehicle_result in (
                vehicle_iterable
            ):

                if not isinstance(
                    vehicle_result,
                    dict,
                ):
                    continue

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

                congestion_values.append(
                    float(congestion)
                )

            if congestion_values:
                average_congestion = (
                    sum(congestion_values)
                    / len(congestion_values)
                )
            else:
                average_congestion = 0.0

            return {
                "valid": True,
                "solution": clone_solution(
                    solution
                ),
                "fitness": float(
                    result["fitness"]
                ),
                "travel_time": (
                    total_travel_time
                ),
                "distance": (
                    total_distance
                ),
                "congestion": (
                    average_congestion
                ),
                "vehicle_results": (
                    vehicle_results
                ),
            }

        except (
            ValueError,
            KeyError,
            TypeError,
        ):

            return {
                "valid": False,
                "solution": None,
                "fitness": float("inf"),
                "travel_time": float("inf"),
                "distance": float("inf"),
                "congestion": float("inf"),
                "vehicle_results": [],
            }

    # ================================================================
    # PARTICLE CREATION
    # ================================================================

    def _create_particle(self):
        """
        Create one particle.
        """

        position = (
            self._random_position()
        )

        velocity = (
            self._random_velocity()
        )

        evaluation = (
            self.evaluate_position(
                position
            )
        )

        fitness = evaluation[
            "fitness"
        ]

        return {
            "position": position,
            "velocity": velocity,
            "fitness": fitness,
            "best_position": list(
                position
            ),
            "best_fitness": fitness,
        }

    # ================================================================
    # INITIALIZE SWARM
    # ================================================================

    def _initialize_particles(self):
        """
        Initialize the swarm.
        """

        return [
            self._create_particle()
            for _ in range(
                self.particles
            )
        ]

    # ================================================================
    # PARTICLE UPDATE
    # ================================================================

    def _update_particle(
        self,
        particle,
        global_best_position,
    ):
        """
        Perform the standard PSO velocity and position update.
        """

        position = particle[
            "position"
        ]

        velocity = particle[
            "velocity"
        ]

        personal_best = particle[
            "best_position"
        ]

        new_velocity = []
        new_position = []

        for i in range(
            self.dimension
        ):

            r1 = self.random.random()
            r2 = self.random.random()

            new_v = (
                self.inertia
                * velocity[i]
                + self.cognitive
                * r1
                * (
                    personal_best[i]
                    - position[i]
                )
                + self.social
                * r2
                * (
                    global_best_position[i]
                    - position[i]
                )
            )

            new_x = (
                position[i]
                + new_v
            )

            # Keep random keys in [0, 1].
            new_x = max(
                0.0,
                min(
                    1.0,
                    new_x,
                ),
            )

            new_velocity.append(
                new_v
            )

            new_position.append(
                new_x
            )

        particle[
            "velocity"
        ] = new_velocity

        particle[
            "position"
        ] = new_position

    # ================================================================
    # GLOBAL BEST
    # ================================================================

    @staticmethod
    def _find_global_best(
        swarm,
    ):
        """
        Return the best feasible particle.
        """

        feasible_particles = [
            particle
            for particle in swarm
            if math.isfinite(
                particle["fitness"]
            )
        ]

        if not feasible_particles:
            return None

        return min(
            feasible_particles,
            key=lambda particle:
                particle["fitness"],
        )

    # ================================================================
    # FALLBACK SOLUTION
    # ================================================================

    def _build_fallback_solution(self):
        """
        Construct a feasible fallback solution.

        First tries the original customer order.
        If necessary, tries descending demand order.
        """

        customer_order = [
            customer.id
            for customer in self.problem.customers
        ]

        solution = (
            self._split_into_vehicle_routes(
                customer_order
            )
        )

        if solution is not None:
            return solution

        # Try descending demand.
        customer_order.sort(
            key=lambda customer_id:
                self.problem.customer_map[
                    customer_id
                ].demand,
            reverse=True,
        )

        return (
            self._split_into_vehicle_routes(
                customer_order
            )
        )

    # ================================================================
    # OPTIMIZATION
    # ================================================================

    def optimize(self):
        """
        Run the PSO optimization.

        Returns
        -------
        dict
            Complete optimization result.
        """

        if not self.problem.is_capacity_feasible():
            raise ValueError(
                "Total vehicle capacity is insufficient "
                "for total demand."
            )

        # ------------------------------------------------------------
        # Initialize swarm.
        # ------------------------------------------------------------

        swarm = (
            self._initialize_particles()
        )

        global_best = (
            self._find_global_best(
                swarm
            )
        )

        # ------------------------------------------------------------
        # If no random particle is feasible, construct a fallback.
        # ------------------------------------------------------------

        if global_best is None:

            fallback_solution = (
                self._build_fallback_solution()
            )

            if fallback_solution is None:
                raise ValueError(
                    "Unable to construct a "
                    "capacity-feasible VRP solution."
                )

            fallback_position = (
                self._random_position()
            )

            fallback_evaluation = (
                self.evaluate_position(
                    fallback_position
                )
            )

            # The fallback position may not represent the fallback
            # solution, so use the actual fallback solution directly.
            fallback_result = (
                self.fitness_function.evaluate(
                    self.problem,
                    fallback_solution,
                )
            )

            global_best_solution = (
                fallback_solution
            )

            global_best_fitness = float(
                fallback_result["fitness"]
            )

            global_best_position = (
                fallback_position
            )

        else:

            global_best_position = list(
                global_best[
                    "best_position"
                ]
            )

            global_best_fitness = float(
                global_best[
                    "best_fitness"
                ]
            )

            global_best_solution = (
                self.decode_position(
                    global_best_position
                )
            )

        # ------------------------------------------------------------
        # Optimization history.
        #
        # Initial fitness + one value for every iteration.
        # ------------------------------------------------------------

        history = [
            global_best_fitness
        ]

        # ------------------------------------------------------------
        # Main PSO loop.
        # ------------------------------------------------------------

        for _ in range(
            self.iterations
        ):

            for particle in swarm:

                self._update_particle(
                    particle,
                    global_best_position,
                )

                evaluation = (
                    self.evaluate_position(
                        particle[
                            "position"
                        ]
                    )
                )

                fitness = evaluation[
                    "fitness"
                ]

                particle[
                    "fitness"
                ] = fitness

                # Personal best.
                if (
                    fitness
                    < particle[
                        "best_fitness"
                    ]
                ):

                    particle[
                        "best_fitness"
                    ] = fitness

                    particle[
                        "best_position"
                    ] = list(
                        particle[
                            "position"
                        ]
                    )

                # Global best.
                if (
                    fitness
                    < global_best_fitness
                ):

                    global_best_fitness = (
                        fitness
                    )

                    global_best_position = (
                        list(
                            particle[
                                "position"
                            ]
                        )
                    )

                    candidate_solution = (
                        evaluation[
                            "solution"
                        ]
                    )

                    if candidate_solution is not None:
                        global_best_solution = (
                            clone_solution(
                                candidate_solution
                            )
                        )

            history.append(
                global_best_fitness
            )

        # ------------------------------------------------------------
        # Ensure final solution exists.
        # ------------------------------------------------------------

        if global_best_solution is None:

            global_best_solution = (
                self.decode_position(
                    global_best_position
                )
            )

        if global_best_solution is None:

            global_best_solution = (
                self._build_fallback_solution()
            )

        if global_best_solution is None:
            raise ValueError(
                "PSO could not produce a "
                "feasible VRP solution."
            )

        # ------------------------------------------------------------
        # Final evaluation.
        # ------------------------------------------------------------

        final_evaluation = (
            self.evaluate_position(
                self._encode_solution_position(
                    global_best_solution
                )
            )
        )

        # Directly evaluate the actual final solution to avoid any
        # mismatch caused by random-key encoding.
        final_result = (
            self.fitness_function.evaluate(
                self.problem,
                global_best_solution,
            )
        )

        vehicle_results = (
            final_result.get(
                "vehicle_results",
                [],
            )
        )

        total_travel_time = 0.0
        total_distance = 0.0
        congestion_values = []

        if isinstance(
            vehicle_results,
            dict,
        ):
            vehicle_iterable = (
                vehicle_results.values()
            )
        else:
            vehicle_iterable = (
                vehicle_results
            )

        for vehicle_result in (
            vehicle_iterable
        ):

            if not isinstance(
                vehicle_result,
                dict,
            ):
                continue

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

            congestion_values.append(
                float(
                    vehicle_result.get(
                        "congestion",
                        0.0,
                    )
                )
            )

        if congestion_values:
            average_congestion = (
                sum(congestion_values)
                / len(congestion_values)
            )
        else:
            average_congestion = 0.0

        return {
            "solution": clone_solution(
                global_best_solution
            ),
            "fitness": float(
                final_result["fitness"]
            ),
            "travel_time": (
                total_travel_time
            ),
            "distance": (
                total_distance
            ),
            "congestion": (
                average_congestion
            ),
            "vehicle_results": (
                vehicle_results
            ),
            "history": history,
            "iterations": self.iterations,
            "particles": self.particles,
            "inertia": self.inertia,
            "cognitive": self.cognitive,
            "social": self.social,
        }

    # ================================================================
    # SOLUTION ENCODING
    # ================================================================

    def _encode_solution_position(
        self,
        solution,
    ):
        """
        Convert a solution back to a random-key position.

        This is mainly used internally when a final solution was
        produced by a fallback repair mechanism.
        """

        ordered_customers = []

        for vehicle in self.problem.vehicles:

            ordered_customers.extend(
                solution.get(
                    vehicle.id,
                    [],
                )
            )

        customer_index = {
            customer_id: index
            for index, customer_id
            in enumerate(
                ordered_customers
            )
        }

        n = len(
            ordered_customers
        )

        if n == 0:
            return [
                0.0
                for _ in range(
                    self.dimension
                )
            ]

        position = []

        for customer in self.problem.customers:

            index = customer_index.get(
                customer.id,
                0,
            )

            position.append(
                (
                    index + 0.5
                ) / n
            )

        return position
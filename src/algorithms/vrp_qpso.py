"""
Quantum-inspired Particle Swarm Optimization for the
Capacitated Vehicle Routing Problem.

Representation
--------------

Each customer receives a continuous random key in [0, 1].
Sorting the keys creates a customer permutation.

The permutation is decoded into capacity-feasible
vehicle routes.

QPSO replaces the classical PSO velocity update with
a quantum-inspired position update.

Lower fitness is better.
"""

import math
import random

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_route_utils import clone_solution


class VRPQPSO(VRPPSO):
    """
    Quantum-inspired Particle Swarm Optimization
    for the Capacitated Vehicle Routing Problem.

    Inherited from VRPPSO:

    - customer random-key representation
    - customer permutation decoding
    - capacity-aware route construction
    - VRP fitness evaluation
    - feasibility checking
    - solution encoding
    - optimization result structure

    QPSO-specific components:

    - personal best position
    - global best position
    - mean best (mbest)
    - quantum-inspired position update
    - beta exploration parameter
    """

    def __init__(
        self,
        problem,
        particles=20,
        iterations=50,
        beta=0.7,
        seed=None,
        fitness_function=None,
    ):
        if problem is None:
            raise ValueError(
                "problem is required."
            )

        if particles < 2:
            raise ValueError(
                "particles must be at least 2."
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be at least 1."
            )

        if beta <= 0:
            raise ValueError(
                "beta must be positive."
            )

        # Reuse VRPPSO for common configuration:
        #
        # - problem
        # - particle count
        # - iteration count
        # - random generator
        # - fitness function
        # - customer dimension
        #
        # Classical PSO velocity parameters are disabled
        # because QPSO does not use velocity.
        super().__init__(
            problem=problem,
            particles=particles,
            iterations=iterations,
            inertia=0.0,
            cognitive=0.0,
            social=0.0,
            seed=seed,
            fitness_function=fitness_function,
        )

        self.beta = float(beta)

    # ================================================================
    # RANDOM POSITION
    # ================================================================

    def _random_position(self):
        """
        Generate a random-key position.

        Each customer receives one continuous value
        in [0, 1].
        """

        return [
            self.random.random()
            for _ in range(self.dimension)
        ]

    # ================================================================
    # QPSO POSITION UPDATE
    # ================================================================

    def _update_position(
        self,
        position,
        best_position,
        mean_best,
        global_best_position=None,
    ):
        """
        Perform the QPSO position update.

        Standard QPSO structure:

            local_attractor =
                phi * personal_best
                + (1 - phi) * global_best

            step =
                beta
                * |mean_best - position|
                * ln(1 / u)

            new_position =
                local_attractor +/- step

        where:

            personal_best = particle's pbest
            global_best   = swarm's gbest
            mean_best     = mbest
            beta          = contraction-expansion coefficient
            u             = random number in (0, 1)

        Random-key values are finally clamped to [0, 1].
        """

        if len(position) != self.dimension:
            raise ValueError(
                "Position dimension does not match "
                "number of customers."
            )

        if len(best_position) != self.dimension:
            raise ValueError(
                "Best-position dimension does not match "
                "number of customers."
            )

        if len(mean_best) != self.dimension:
            raise ValueError(
                "Mean-best dimension does not match "
                "number of customers."
            )

        # Backward compatibility:
        #
        # If global_best_position is not supplied, use the
        # personal best. This keeps direct unit tests of the
        # older three-argument method functional.
        if global_best_position is None:
            global_best_position = best_position

        if len(global_best_position) != self.dimension:
            raise ValueError(
                "Global-best dimension does not match "
                "number of customers."
            )

        new_position = []

        for i in range(self.dimension):

            # --------------------------------------------------------
            # Random value for logarithmic quantum step.
            # Avoid log(0).
            # --------------------------------------------------------

            u = max(
                self.random.random(),
                1e-12,
            )

            # --------------------------------------------------------
            # Random attractor coefficient.
            # --------------------------------------------------------

            phi = self.random.random()

            # --------------------------------------------------------
            # Standard QPSO local attractor:
            #
            # p = phi * pbest
            #     + (1 - phi) * gbest
            # --------------------------------------------------------

            local_attractor = (
                phi * best_position[i]
                + (1.0 - phi)
                * global_best_position[i]
            )

            # --------------------------------------------------------
            # Quantum step.
            #
            # mbest determines the characteristic
            # swarm spread.
            # --------------------------------------------------------

            step = (
                self.beta
                * abs(
                    mean_best[i]
                    - position[i]
                )
                * math.log(1.0 / u)
            )

            # --------------------------------------------------------
            # QPSO has two possible directions.
            # --------------------------------------------------------

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

            # --------------------------------------------------------
            # Random-key values must remain in [0, 1].
            # --------------------------------------------------------

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

    # ================================================================
    # MEAN BEST
    # ================================================================

    @staticmethod
    def _calculate_mean_best(
        swarm,
        dimension,
    ):
        """
        Calculate the mean of all particles'
        personal-best positions.

        This is the QPSO mbest vector.
        """

        if not swarm:
            return [
                0.5
                for _ in range(dimension)
            ]

        mean_best = []

        for i in range(dimension):

            total = sum(
                particle["best_position"][i]
                for particle in swarm
            )

            mean_best.append(
                total / len(swarm)
            )

        return mean_best

    # Backward-compatible alias.
    _mean_best = _calculate_mean_best

    # ================================================================
    # PARTICLE CREATION
    # ================================================================

    def _create_particle(self):
        """
        Create one QPSO particle.

        Unlike classical PSO, QPSO does not require
        a velocity vector.
        """

        position = self._random_position()

        evaluation = self.evaluate_position(
            position
        )

        fitness = evaluation["fitness"]

        return {
            "position": list(position),
            "fitness": fitness,
            "best_position": list(position),
            "best_fitness": fitness,
        }

    # ================================================================
    # INITIALIZE SWARM
    # ================================================================

    def _initialize_particles(self):
        """
        Initialize the QPSO swarm.
        """

        return [
            self._create_particle()
            for _ in range(self.particles)
        ]

    # ================================================================
    # GLOBAL BEST
    # ================================================================

    @staticmethod
    def _find_global_best(swarm):
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
    # OPTIMIZATION
    # ================================================================

    def optimize(self):
        """
        Run QPSO optimization.

        Returns
        -------

        dict
            Complete VRP optimization result.

        Result fields
        -------------

        valid
        solution
        fitness
        travel_time
        distance
        congestion
        vehicle_results
        history
        iterations
        particles
        beta
        """

        # ------------------------------------------------------------
        # Capacity feasibility
        # ------------------------------------------------------------

        if not self.problem.is_capacity_feasible():
            return {
                "valid": False,
                "solution": None,
                "fitness": float("inf"),
                "travel_time": float("inf"),
                "distance": float("inf"),
                "congestion": float("inf"),
                "vehicle_results": [],
                "history": [],
                "iterations": self.iterations,
                "particles": self.particles,
                "beta": self.beta,
            }

        # ------------------------------------------------------------
        # Initialize swarm
        # ------------------------------------------------------------

        swarm = self._initialize_particles()

        global_best_particle = (
            self._find_global_best(swarm)
        )

        global_best_position = None
        global_best_fitness = float("inf")

        if global_best_particle is not None:

            global_best_position = list(
                global_best_particle[
                    "best_position"
                ]
            )

            global_best_fitness = (
                global_best_particle[
                    "best_fitness"
                ]
            )

        # ------------------------------------------------------------
        # Fallback solution
        #
        # This guarantees that a feasible solution exists
        # whenever the problem itself is capacity feasible.
        # ------------------------------------------------------------

        if global_best_position is None:

            fallback_solution = (
                self._build_fallback_solution()
            )

            fallback_evaluation = (
                self.fitness_function.evaluate(
                    self.problem,
                    fallback_solution,
                )
            )

            global_best_fitness = (
                fallback_evaluation["fitness"]
            )

            global_best_position = (
                self._encode_solution_position(
                    fallback_solution
                )
            )

        # ------------------------------------------------------------
        # Convergence history
        #
        # Includes the initial solution.
        # Therefore final length = iterations + 1.
        # ------------------------------------------------------------

        history = [
            global_best_fitness
        ]

        # ------------------------------------------------------------
        # Main QPSO loop
        # ------------------------------------------------------------

        for _ in range(self.iterations):

            # Calculate mbest from all personal best positions.
            mean_best = (
                self._calculate_mean_best(
                    swarm,
                    self.dimension,
                )
            )

            for particle in swarm:

                current_position = list(
                    particle["position"]
                )

                personal_best_position = list(
                    particle["best_position"]
                )

                # ----------------------------------------------------
                # QPSO movement:
                #
                # pbest + gbest attractor
                # mbest quantum step
                # ----------------------------------------------------

                new_position = (
                    self._update_position(
                        position=current_position,
                        best_position=(
                            personal_best_position
                        ),
                        mean_best=mean_best,
                        global_best_position=(
                            global_best_position
                        ),
                    )
                )

                evaluation = (
                    self.evaluate_position(
                        new_position
                    )
                )

                new_fitness = (
                    evaluation["fitness"]
                )

                particle["position"] = list(
                    new_position
                )

                particle["fitness"] = (
                    new_fitness
                )

                # ----------------------------------------------------
                # Personal best update
                # ----------------------------------------------------

                if new_fitness < particle[
                    "best_fitness"
                ]:

                    particle["best_position"] = (
                        list(new_position)
                    )

                    particle["best_fitness"] = (
                        new_fitness
                    )

                # ----------------------------------------------------
                # Global best update
                # ----------------------------------------------------

                if new_fitness < global_best_fitness:

                    global_best_fitness = (
                        new_fitness
                    )

                    global_best_position = (
                        list(new_position)
                    )

                elif (
                    particle["best_fitness"]
                    < global_best_fitness
                ):

                    global_best_fitness = (
                        particle["best_fitness"]
                    )

                    global_best_position = (
                        list(
                            particle[
                                "best_position"
                            ]
                        )
                    )

            # --------------------------------------------------------
            # Store best fitness after this iteration.
            # --------------------------------------------------------

            history.append(
                global_best_fitness
            )

        # ------------------------------------------------------------
        # Decode final global best
        # ------------------------------------------------------------

        final_solution = (
            self.decode_position(
                global_best_position
            )
        )

        # Safety fallback.
        if final_solution is None:

            final_solution = (
                self._build_fallback_solution()
            )

        # ------------------------------------------------------------
        # Evaluate final solution using the same VRP fitness
        # function used throughout the optimization.
        # ------------------------------------------------------------

        final_evaluation = (
            self.fitness_function.evaluate(
                self.problem,
                final_solution,
            )
        )

        final_fitness = (
            final_evaluation["fitness"]
        )

        vehicle_results = (
            final_evaluation.get(
                "vehicle_results",
                [],
            )
        )

        # ------------------------------------------------------------
        # Aggregate vehicle metrics.
        # ------------------------------------------------------------

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

            travel_time = (
                vehicle_result.get(
                    "travel_time",
                    0.0,
                )
            )

            distance = (
                vehicle_result.get(
                    "distance",
                    0.0,
                )
            )

            congestion = (
                vehicle_result.get(
                    "congestion",
                    0.0,
                )
            )

            if math.isfinite(
                float(travel_time)
            ):
                total_travel_time += (
                    float(travel_time)
                )

            if math.isfinite(
                float(distance)
            ):
                total_distance += (
                    float(distance)
                )

            if math.isfinite(
                float(congestion)
            ):
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

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        return {
            "valid": True,
            "solution": clone_solution(
                final_solution
            ),
            "fitness": final_fitness,
            "travel_time": total_travel_time,
            "distance": total_distance,
            "congestion": average_congestion,
            "vehicle_results": vehicle_results,
            "history": history,
            "iterations": self.iterations,
            "particles": self.particles,
            "beta": self.beta,
        }
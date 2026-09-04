"""
Warm-Start Quantum Particle Swarm Optimization for Vehicle Routing.

Architecture
------------

VRP
 |
 +--> Short classical PSO warm-start
 |
 +--> Encode warm solution as random keys
 |
 +--> Inject warm particle into QPSO swarm
 |
 +--> Quantum-inspired particle updates
 |
 +--> Best VRP solution

Optional solution recording
---------------------------

When `record_solutions=True`, the optimizer records the best complete
VRP solution at iteration 0 and after every QPSO iteration.

This is used by Decomposed Warm-QPSO to construct a globally
comparable convergence curve.
"""

import time

from src.algorithms.vrp_pso import VRPPSO
from src.algorithms.vrp_qpso import VRPQPSO


class WarmVRPQPSO(VRPQPSO):
    """
    Warm-start QPSO for the Capacitated Vehicle Routing Problem.

    A short classical PSO run generates a warm solution. That solution
    is encoded as a random-key position and injected as the first
    particle of the QPSO swarm.

    Parameters
    ----------
    problem:
        VRPProblem instance.

    particles:
        Number of QPSO particles.

    iterations:
        Number of QPSO iterations.

    beta:
        QPSO contraction-expansion coefficient.

    seed:
        Random seed.

    fitness_function:
        Optional VRPFitnessFunction.

    warm_iterations:
        Number of classical PSO iterations used for warm-start.

    warm_particles:
        Number of classical PSO particles used for warm-start.

    record_solutions:
        If True, store the best complete VRP solution at every
        iteration in `solution_history`.
    """

    def __init__(
        self,
        problem,
        particles=20,
        iterations=50,
        beta=0.7,
        seed=None,
        fitness_function=None,
        warm_iterations=10,
        warm_particles=10,
        record_solutions=False,
    ):
        super().__init__(
            problem=problem,
            particles=particles,
            iterations=iterations,
            beta=beta,
            seed=seed,
            fitness_function=fitness_function,
        )

        # Keep the seed explicitly.
        self.seed = seed

        if warm_iterations < 1:
            raise ValueError(
                "warm_iterations must be at least 1."
            )

        if warm_particles < 2:
            raise ValueError(
                "warm_particles must be at least 2."
            )

        self.warm_iterations = int(
            warm_iterations
        )

        self.warm_particles = int(
            warm_particles
        )

        self.record_solutions = bool(
            record_solutions
        )

    # ================================================================
    # WARM START
    # ================================================================

    def _generate_warm_solution(self):
        """
        Run a short classical PSO optimization to obtain a warm start.

        Important:
        We only require that PSO returns a solution.

        We intentionally do NOT reject the result solely because its
        `valid` flag is False. This preserves the original Warm-QPSO
        behavior and allows the returned solution to be used as the
        initialization for QPSO.
        """

        if self.seed is None:
            warm_seed = 10000
        else:
            warm_seed = int(self.seed) + 10000

        warm_optimizer = VRPPSO(
            problem=self.problem,
            particles=self.warm_particles,
            iterations=self.warm_iterations,
            seed=warm_seed,
            fitness_function=self.fitness_function,
        )

        result = warm_optimizer.optimize()

        if result.get("solution") is None:
            raise RuntimeError(
                "Warm-start PSO returned no solution."
            )

        return result

    # ================================================================
    # SOLUTION -> RANDOM KEY POSITION
    # ================================================================

    def _solution_to_position(self, solution):
        """
        Convert a VRP solution into a random-key position.

        Customers are ordered according to vehicle order and route
        order. That ordering is then converted into keys in (0, 1).

        The position uses the same customer ordering as the VRP
        random-key representation.
        """

        customer_order = []

        # ------------------------------------------------------------
        # Collect customers according to vehicle and route order.
        # ------------------------------------------------------------

        for vehicle_id in sorted(
            solution.keys(),
            key=str,
        ):
            route = solution[vehicle_id]

            for customer_id in route:
                customer_id = str(
                    customer_id
                )

                if (
                    customer_id
                    in self.problem.customer_map
                ):
                    customer_order.append(
                        customer_id
                    )

        # ------------------------------------------------------------
        # Add missing customers if necessary.
        # ------------------------------------------------------------

        existing = set(
            customer_order
        )

        missing = [
            str(customer.id)
            for customer in self.problem.customers
            if str(customer.id)
            not in existing
        ]

        customer_order.extend(
            missing
        )

        n = len(
            customer_order
        )

        if n != self.dimension:
            raise ValueError(
                "Encoded solution dimension does not match "
                "number of customers."
            )

        # ------------------------------------------------------------
        # Assign deterministic ranks.
        # ------------------------------------------------------------

        rank = {
            customer_id: index
            for index, customer_id
            in enumerate(customer_order)
        }

        # ------------------------------------------------------------
        # Convert ranks to random keys.
        # ------------------------------------------------------------

        position = []

        for customer in self.problem.customers:

            customer_id = str(
                customer.id
            )

            customer_rank = rank[
                customer_id
            ]

            key = (
                customer_rank + 1
            ) / (
                n + 1
            )

            position.append(
                key
            )

        return position

    # Backward-compatible alias.
    def _solution_to_position_keys(
        self,
        solution,
    ):
        return self._solution_to_position(
            solution
        )

    # ================================================================
    # CREATE WARM SWARM
    # ================================================================

    def _create_warm_swarm(
        self,
        warm_position,
    ):
        """
        Create the initial QPSO swarm.

        The warm-start particle is inserted first.

        Remaining particles are randomly initialized.
        """

        swarm = []

        # ------------------------------------------------------------
        # Warm particle.
        # ------------------------------------------------------------

        warm_result = (
            self.evaluate_position(
                warm_position
            )
        )

        swarm.append(
            {
                "position": (
                    warm_position[:]
                ),

                "best_position": (
                    warm_position[:]
                ),

                "best_fitness": (
                    warm_result["fitness"]
                ),
            }
        )

        # ------------------------------------------------------------
        # Random particles.
        # ------------------------------------------------------------

        while len(swarm) < self.particles:

            position = (
                self._random_position()
            )

            result = (
                self.evaluate_position(
                    position
                )
            )

            swarm.append(
                {
                    "position": position,

                    "best_position": (
                        position[:]
                    ),

                    "best_fitness": (
                        result["fitness"]
                    ),
                }
            )

        return swarm

    # ================================================================
    # COPY SOLUTION
    # ================================================================

    @staticmethod
    def _copy_solution(
        solution,
    ):
        """
        Make an independent copy of a VRP solution.

        A VRP solution has the form:

            {
                "V1": ["C1", "C4"],
                "V2": ["C2", "C3"]
            }
        """

        if solution is None:
            return None

        return {
            str(vehicle_id): [
                str(customer_id)
                for customer_id in route
            ]
            for vehicle_id, route
            in solution.items()
        }

    # ================================================================
    # OPTIMIZATION
    # ================================================================

    def optimize(self):
        """
        Execute Warm-QPSO.

        Returns
        -------
        dict
            Complete VRP optimization result.

        Standard fields
        ---------------
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
        warm_iterations
        warm_particles
        warm_start_fitness
        execution_time

        Additional field
        ----------------
        solution_history

            Only included when:

                record_solutions=True

            Contains the best complete VRP solution at:

                iteration 0
                iteration 1
                ...
                iteration N
        """

        start_time = time.perf_counter()

        # ============================================================
        # STEP 1
        # GENERATE CLASSICAL PSO WARM START
        # ============================================================

        warm_result = (
            self._generate_warm_solution()
        )

        warm_position = (
            self._solution_to_position(
                warm_result["solution"]
            )
        )

        # ============================================================
        # STEP 2
        # CREATE INITIAL QPSO SWARM
        # ============================================================

        swarm = (
            self._create_warm_swarm(
                warm_position
            )
        )

        # ============================================================
        # STEP 3
        # INITIAL GLOBAL BEST
        # ============================================================

        global_best = min(
            swarm,
            key=lambda particle:
            particle["best_fitness"],
        )

        global_best_position = (
            global_best[
                "best_position"
            ][:]
        )

        global_best_fitness = (
            global_best[
                "best_fitness"
            ]
        )

        # ============================================================
        # FITNESS HISTORY
        # ============================================================

        history = [
            global_best_fitness
        ]

        # ============================================================
        # SOLUTION HISTORY
        # ============================================================

        solution_history = []

        if self.record_solutions:

            initial_result = (
                self.evaluate_position(
                    global_best_position
                )
            )

            solution_history.append(
                self._copy_solution(
                    initial_result[
                        "solution"
                    ]
                )
            )

        # ============================================================
        # STEP 4
        # QPSO ITERATIONS
        # ============================================================

        for _ in range(
            self.iterations
        ):

            # --------------------------------------------------------
            # Calculate mean best position.
            # --------------------------------------------------------

            mean_best = (
                self._calculate_mean_best(
                    swarm,
                    self.dimension,
                )
            )

            # --------------------------------------------------------
            # Keep a snapshot of the current global best.
            # --------------------------------------------------------

            iteration_global_best = (
                global_best_position[:]
            )

            # --------------------------------------------------------
            # Update every particle.
            # --------------------------------------------------------

            for particle in swarm:

                new_position = (
                    self._update_position(
                        particle[
                            "position"
                        ],

                        particle[
                            "best_position"
                        ],

                        mean_best,

                        iteration_global_best,
                    )
                )

                result = (
                    self.evaluate_position(
                        new_position
                    )
                )

                particle[
                    "position"
                ] = new_position

                # ----------------------------------------------------
                # Update personal best.
                # ----------------------------------------------------

                if (
                    result["fitness"]
                    <
                    particle[
                        "best_fitness"
                    ]
                ):

                    particle[
                        "best_fitness"
                    ] = (
                        result[
                            "fitness"
                        ]
                    )

                    particle[
                        "best_position"
                    ] = (
                        new_position[:]
                    )

                    # ------------------------------------------------
                    # Update global best.
                    # ------------------------------------------------

                    if (
                        particle[
                            "best_fitness"
                        ]
                        <
                        global_best_fitness
                    ):

                        global_best_fitness = (
                            particle[
                                "best_fitness"
                            ]
                        )

                        global_best_position = (
                            particle[
                                "best_position"
                            ][:]
                        )

            # --------------------------------------------------------
            # Record best-so-far fitness.
            # --------------------------------------------------------

            history.append(
                global_best_fitness
            )

            # --------------------------------------------------------
            # Record best-so-far complete VRP solution.
            # --------------------------------------------------------

            if self.record_solutions:

                iteration_result = (
                    self.evaluate_position(
                        global_best_position
                    )
                )

                solution_history.append(
                    self._copy_solution(
                        iteration_result[
                            "solution"
                        ]
                    )
                )

        # ============================================================
        # STEP 5
        # FINAL EVALUATION
        # ============================================================

        final_result = (
            self.evaluate_position(
                global_best_position
            )
        )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        # ============================================================
        # STEP 6
        # BUILD RESULT
        # ============================================================

        result = {
            "valid": final_result[
                "valid"
            ],

            "solution": final_result[
                "solution"
            ],

            "fitness": final_result[
                "fitness"
            ],

            "travel_time": final_result[
                "travel_time"
            ],

            "distance": final_result[
                "distance"
            ],

            "congestion": final_result[
                "congestion"
            ],

            "vehicle_results": final_result[
                "vehicle_results"
            ],

            "history": history,

            "iterations": self.iterations,

            "particles": self.particles,

            "beta": self.beta,

            "warm_iterations": (
                self.warm_iterations
            ),

            "warm_particles": (
                self.warm_particles
            ),

            "warm_start_fitness": (
                warm_result[
                    "fitness"
                ]
            ),

            "execution_time": (
                execution_time
            ),
        }

        # ============================================================
        # OPTIONAL SOLUTION HISTORY
        # ============================================================

        if self.record_solutions:

            result[
                "solution_history"
            ] = solution_history

        return result
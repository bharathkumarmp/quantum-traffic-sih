"""
Decomposed Warm-QPSO for Large-Scale Vehicle Routing Problems.

Architecture
------------

Large VRP
   |
   v
Customer decomposition
   |
   v
Capacity-aware vehicle allocation
   |
   v
Warm-QPSO on each cluster
   |
   v
Merge cluster solutions
   |
   v
Fast global boundary repair
   |
   v
Final VRP solution


Important
---------
The current clustering strategy is deterministic and demand-aware.

It uses customer node ordering as a lightweight proxy for spatial
locality because the current synthetic graph does not contain
geographical coordinates.

For a production smart-city system, this can later be replaced by
network-distance or traffic-aware clustering.
"""

import random
import time

from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.domain.vrp_problem import VRPProblem
from src.optimization.vrp_fitness import VRPFitnessFunction
from src.optimization.global_repair import GlobalBoundaryRepair


class DecomposedWarmQPSO:
    """
    Decomposition-based Warm-QPSO for large-scale VRP.
    """

    def __init__(
        self,
        problem,
        clusters=5,
        particles=20,
        iterations=50,
        beta=0.7,
        seed=None,
        fitness_function=None,
        warm_iterations=10,
        warm_particles=10,
        repair=True,
        repair_rounds=2,
        max_boundary_customers=20,
        max_target_vehicles=5,
        try_swaps=True,
    ):
        # ============================================================
        # VALIDATION
        # ============================================================

        if problem is None:
            raise ValueError(
                "problem is required."
            )

        if clusters < 1:
            raise ValueError(
                "clusters must be at least 1."
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

        if warm_iterations < 1:
            raise ValueError(
                "warm_iterations must be at least 1."
            )

        if warm_particles < 2:
            raise ValueError(
                "warm_particles must be at least 2."
            )

        if repair_rounds < 1:
            raise ValueError(
                "repair_rounds must be at least 1."
            )

        if max_boundary_customers < 1:
            raise ValueError(
                "max_boundary_customers must be at least 1."
            )

        if max_target_vehicles < 1:
            raise ValueError(
                "max_target_vehicles must be at least 1."
            )

        # ============================================================
        # STORE CONFIGURATION
        # ============================================================

        self.problem = problem

        self.clusters = min(
            int(clusters),
            problem.number_of_customers,
        )

        self.particles = int(
            particles
        )

        self.iterations = int(
            iterations
        )

        self.beta = float(
            beta
        )

        self.seed = seed

        self.random = random.Random(
            seed
        )

        self.fitness_function = (
            fitness_function
            if fitness_function is not None
            else VRPFitnessFunction()
        )

        self.warm_iterations = int(
            warm_iterations
        )

        self.warm_particles = int(
            warm_particles
        )

        self.repair = bool(
            repair
        )

        self.repair_rounds = int(
            repair_rounds
        )

        self.max_boundary_customers = int(
            max_boundary_customers
        )

        self.max_target_vehicles = int(
            max_target_vehicles
        )

        self.try_swaps = bool(
            try_swaps
        )

    # ================================================================
    # CUSTOMER FEATURES
    # ================================================================

    def _customer_features(self):
        """
        Generate deterministic customer features.

        Because the current synthetic graph does not contain geographic
        coordinates, node ordering is used as a lightweight locality
        proxy.

        Returns
        -------
        list[tuple]
            Each tuple contains:

                (
                    customer,
                    numeric_node_order,
                    demand
                )
        """

        # ------------------------------------------------------------
        # Build deterministic node ordering.
        # ------------------------------------------------------------

        node_ids = [
            str(node.id)
            for node in self.problem.graph.nodes
        ]

        node_order = {
            node_id: index
            for index, node_id
            in enumerate(node_ids)
        }

        features = []

        for customer in self.problem.customers:

            node_id = str(
                customer.node_id
            )

            position = node_order.get(
                node_id,
                0,
            )

            features.append(
                (
                    customer,
                    position,
                    float(customer.demand),
                )
            )

        return features

    # ================================================================
    # CREATE CLUSTERS
    # ================================================================

    def _create_clusters(self):
        """
        Create deterministic demand-aware clusters.

        Customers are ordered using their node position and then
        distributed using a demand-balanced strategy.
        """

        features = (
            self._customer_features()
        )

        if not features:
            return []

        # ------------------------------------------------------------
        # Sort using node position.
        # ------------------------------------------------------------

        features.sort(
            key=lambda item: (
                item[1],
                item[0].id,
            )
        )

        number_of_clusters = min(
            self.clusters,
            len(features),
        )

        clusters = [
            []
            for _ in range(number_of_clusters)
        ]

        cluster_demands = [
            0.0
            for _ in range(number_of_clusters)
        ]

        # ------------------------------------------------------------
        # Greedy balanced assignment.
        # ------------------------------------------------------------

        for customer, _, demand in features:

            target_cluster = min(
                range(number_of_clusters),
                key=lambda index:
                cluster_demands[index],
            )

            clusters[
                target_cluster
            ].append(customer)

            cluster_demands[
                target_cluster
            ] += demand

        # ------------------------------------------------------------
        # Remove empty clusters.
        # ------------------------------------------------------------

        clusters = [
            cluster
            for cluster in clusters
            if cluster
        ]

        return clusters

    # ================================================================
    # ALLOCATE VEHICLES
    # ================================================================

    def _allocate_vehicles(
        self,
        clusters,
    ):
        """
        Allocate vehicles to clusters.

        Every cluster receives at least one vehicle.

        Remaining vehicles are assigned according to:

            cluster demand / allocated capacity
        """

        vehicles = list(
            self.problem.vehicles
        )

        if len(vehicles) < len(clusters):
            raise ValueError(
                "The number of vehicles must be "
                "at least the number of clusters."
            )

        # ------------------------------------------------------------
        # Cluster demands.
        # ------------------------------------------------------------

        cluster_demands = [
            sum(
                customer.demand
                for customer in cluster
            )
            for cluster in clusters
        ]

        # ------------------------------------------------------------
        # Vehicle allocations.
        # ------------------------------------------------------------

        allocations = [
            []
            for _ in clusters
        ]

        remaining_vehicles = vehicles[:]

        # ------------------------------------------------------------
        # First give every cluster one vehicle.
        # ------------------------------------------------------------

        for index in range(
            len(clusters)
        ):
            allocations[index].append(
                remaining_vehicles.pop(0)
            )

        # ------------------------------------------------------------
        # Allocate remaining vehicles.
        # ------------------------------------------------------------

        while remaining_vehicles:

            def load_ratio(index):

                allocated_capacity = sum(
                    vehicle.capacity
                    for vehicle
                    in allocations[index]
                )

                if allocated_capacity <= 0:
                    return float("inf")

                return (
                    cluster_demands[index]
                    / allocated_capacity
                )

            best_cluster = max(
                range(len(clusters)),
                key=load_ratio,
            )

            allocations[
                best_cluster
            ].append(
                remaining_vehicles.pop(0)
            )

        return allocations

    # ================================================================
    # BUILD SUBPROBLEM
    # ================================================================

    def _build_subproblem(
        self,
        customers,
        vehicles,
    ):
        """
        Build a VRP subproblem for one cluster.

        All subproblems share the original transportation graph.
        """

        return VRPProblem(
            graph=self.problem.graph,
            depot=self.problem.depot,
            customers=list(customers),
            vehicles=list(vehicles),
        )

    # ================================================================
    # SOLVE CLUSTERS
    # ================================================================

    def _solve_clusters(
        self,
        clusters,
        vehicle_allocations,
    ):
        """
        Solve every cluster independently using Warm-QPSO.

        `record_solutions=True` is important here because the global
        convergence history is reconstructed from the best complete
        solution of every cluster at every iteration.
        """

        cluster_results = []

        for index, customers in enumerate(
            clusters
        ):

            vehicles = (
                vehicle_allocations[index]
            )

            # --------------------------------------------------------
            # Build cluster problem.
            # --------------------------------------------------------

            subproblem = (
                self._build_subproblem(
                    customers,
                    vehicles,
                )
            )

            # --------------------------------------------------------
            # Deterministic cluster seed.
            # --------------------------------------------------------

            if self.seed is None:
                cluster_seed = (
                    1000 + index
                )
            else:
                cluster_seed = (
                    int(self.seed)
                    + 1000
                    + index
                )

            # --------------------------------------------------------
            # Create Warm-QPSO.
            # --------------------------------------------------------

            optimizer = WarmVRPQPSO(
                problem=subproblem,
                particles=self.particles,
                iterations=self.iterations,
                beta=self.beta,
                seed=cluster_seed,
                fitness_function=self.fitness_function,
                warm_iterations=self.warm_iterations,
                warm_particles=self.warm_particles,

                # IMPORTANT:
                # Record the best solution at every iteration.
                record_solutions=True,
            )

            # --------------------------------------------------------
            # Optimize cluster.
            # --------------------------------------------------------

            result = optimizer.optimize()

            if not result.get(
                "valid",
                False,
            ):
                raise RuntimeError(
                    "Warm-QPSO produced an invalid "
                    f"solution for cluster {index + 1}."
                )

            if result.get(
                "solution"
            ) is None:
                raise RuntimeError(
                    "Warm-QPSO returned no solution "
                    f"for cluster {index + 1}."
                )

            solution_history = result.get(
                "solution_history"
            )

            if not solution_history:
                raise RuntimeError(
                    "Warm-QPSO did not return a "
                    f"solution history for cluster {index + 1}."
                )

            # --------------------------------------------------------
            # Store cluster information.
            # --------------------------------------------------------

            cluster_results.append(
                {
                    "cluster_id": index + 1,

                    "customers": [
                        str(customer.id)
                        for customer in customers
                    ],

                    "vehicles": [
                        str(vehicle.id)
                        for vehicle in vehicles
                    ],

                    "result": result,

                    # Explicit copy for easier access.
                    "solution_history": [
                        self._copy_solution(
                            solution
                        )
                        for solution
                        in solution_history
                    ],
                }
            )

        return cluster_results

    # ================================================================
    # COPY SOLUTION
    # ================================================================

    @staticmethod
    def _copy_solution(solution):
        """
        Copy a VRP solution.
        """

        return {
            str(vehicle_id): [
                str(customer_id)
                for customer_id in route
            ]
            for vehicle_id, route
            in solution.items()
        }

    # ================================================================
    # MERGE CLUSTER SOLUTIONS
    # ================================================================

    def _merge_solutions(
        self,
        cluster_results,
    ):
        """
        Merge all cluster-level solutions.

        Since vehicles are allocated exclusively to clusters,
        duplicate vehicle assignments should not occur.
        """

        merged_solution = {}

        for cluster in cluster_results:

            solution = cluster[
                "result"
            ]["solution"]

            if solution is None:
                raise ValueError(
                    "Cluster returned no solution."
                )

            for vehicle_id, route in (
                solution.items()
            ):

                vehicle_id = str(
                    vehicle_id
                )

                if vehicle_id in merged_solution:
                    raise ValueError(
                        "Duplicate vehicle assignment: "
                        f"{vehicle_id}"
                    )

                merged_solution[
                    vehicle_id
                ] = [
                    str(customer_id)
                    for customer_id in route
                ]

        return merged_solution

    # ================================================================
    # MERGE SOLUTION SNAPSHOTS
    # ================================================================

    def _merge_solution_histories(
        self,
        cluster_results,
    ):
        """
        Construct merged global solutions for every iteration.

        Every cluster provides:

            solution_history[0]
            solution_history[1]
            ...
            solution_history[N]

        At each iteration, the corresponding cluster solutions are
        merged into one complete VRP solution.
        """

        if not cluster_results:
            return []

        history_lengths = [
            len(
                cluster.get(
                    "solution_history",
                    [],
                )
            )
            for cluster in cluster_results
        ]

        if not history_lengths:
            return []

        if any(
            length == 0
            for length in history_lengths
        ):
            raise RuntimeError(
                "At least one cluster has an empty "
                "solution history."
            )

        # Use the shortest history so that every cluster contributes
        # a valid snapshot.
        history_length = min(
            history_lengths
        )

        global_solution_history = []

        for iteration in range(
            history_length
        ):

            merged_solution = {}

            for cluster in cluster_results:

                cluster_history = cluster[
                    "solution_history"
                ]

                cluster_solution = (
                    cluster_history[
                        iteration
                    ]
                )

                for vehicle_id, route in (
                    cluster_solution.items()
                ):

                    vehicle_id = str(
                        vehicle_id
                    )

                    if vehicle_id in merged_solution:
                        raise ValueError(
                            "Duplicate vehicle assignment "
                            "during convergence merge: "
                            f"{vehicle_id}"
                        )

                    merged_solution[
                        vehicle_id
                    ] = [
                        str(customer_id)
                        for customer_id in route
                    ]

            global_solution_history.append(
                merged_solution
            )

        return global_solution_history

    # ================================================================
    # GLOBAL CONVERGENCE HISTORY
    # ================================================================

    def _build_global_convergence_history(
        self,
        cluster_results,
    ):
        """
        Build a globally comparable convergence history.

        IMPORTANT
        ---------
        This does NOT sum cluster fitness values.

        Instead:

            cluster solution snapshots
                    ↓
              merge complete VRP
                    ↓
          evaluate original VRP
                    ↓
             global fitness

        Therefore the resulting history is directly comparable with
        PSO, QPSO and Warm-QPSO histories.
        """

        solution_history = (
            self._merge_solution_histories(
                cluster_results
            )
        )

        history = []

        for iteration, solution in enumerate(
            solution_history
        ):

            result = (
                self._evaluate_solution(
                    solution
                )
            )

            if not result.get(
                "valid",
                False,
            ):
                raise RuntimeError(
                    "Merged convergence solution "
                    f"at iteration {iteration} is invalid."
                )

            history.append(
                float(result["fitness"])
            )

        return history

    # ================================================================
    # EVALUATE COMPLETE SOLUTION
    # ================================================================

    def _evaluate_solution(
        self,
        solution,
    ):
        """
        Evaluate the complete solution using the original problem.
        """

        return self.fitness_function.evaluate(
            problem=self.problem,
            solution=solution,
        )

    # ================================================================
    # VALIDATE FINAL SOLUTION
    # ================================================================

    def _validate_final_solution(
        self,
        solution,
    ):
        """
        Validate the final merged/repaired solution.
        """

        expected_vehicle_ids = {
            str(vehicle.id)
            for vehicle in self.problem.vehicles
        }

        actual_vehicle_ids = {
            str(vehicle_id)
            for vehicle_id in solution.keys()
        }

        if actual_vehicle_ids != expected_vehicle_ids:
            return False

        expected_customer_ids = {
            str(customer.id)
            for customer in self.problem.customers
        }

        assigned_customers = []

        for route in solution.values():
            assigned_customers.extend(
                str(customer_id)
                for customer_id in route
            )

        # No duplicate customer.
        if len(assigned_customers) != len(
            set(assigned_customers)
        ):
            return False

        # Exact customer coverage.
        return (
            set(assigned_customers)
            == expected_customer_ids
        )

    # ================================================================
    # CLUSTER SUMMARY
    # ================================================================

    def _build_cluster_summary(
        self,
        cluster_results,
    ):
        """
        Create compact cluster-level diagnostics.
        """

        summary = []

        for cluster in cluster_results:

            result = cluster[
                "result"
            ]

            summary.append(
                {
                    "cluster_id": cluster[
                        "cluster_id"
                    ],

                    "customers": cluster[
                        "customers"
                    ],

                    "vehicles": cluster[
                        "vehicles"
                    ],

                    "fitness": result[
                        "fitness"
                    ],

                    "travel_time": result[
                        "travel_time"
                    ],

                    "distance": result[
                        "distance"
                    ],

                    "congestion": result[
                        "congestion"
                    ],

                    "warm_start_fitness": (
                        result.get(
                            "warm_start_fitness"
                        )
                    ),

                    "execution_time": (
                        result.get(
                            "execution_time"
                        )
                    ),

                    "history": result.get(
                        "history",
                        [],
                    ),

                    "solution_history_length": len(
                        cluster.get(
                            "solution_history",
                            [],
                        )
                    ),
                }
            )

        return summary

    # ================================================================
    # OPTIMIZATION
    # ================================================================

    def optimize(self):
        """
        Execute the complete decomposed Warm-QPSO pipeline.

        Pipeline
        --------
        1. Create customer clusters.
        2. Allocate vehicles.
        3. Solve every cluster using Warm-QPSO.
        4. Build globally comparable convergence history.
        5. Merge final cluster solutions.
        6. Evaluate merged solution.
        7. Apply optional global boundary repair.
        8. Validate final solution.
        9. Return final metrics.
        """

        start_time = time.perf_counter()

        # ============================================================
        # STEP 1: CREATE CLUSTERS
        # ============================================================

        clusters = (
            self._create_clusters()
        )

        if not clusters:
            raise RuntimeError(
                "Customer decomposition produced "
                "no clusters."
            )

        # ============================================================
        # STEP 2: ALLOCATE VEHICLES
        # ============================================================

        vehicle_allocations = (
            self._allocate_vehicles(
                clusters
            )
        )

        # ============================================================
        # STEP 3: SOLVE EACH CLUSTER
        # ============================================================

        cluster_results = (
            self._solve_clusters(
                clusters,
                vehicle_allocations,
            )
        )

        # ============================================================
        # STEP 4: GLOBAL CONVERGENCE HISTORY
        # ============================================================

        convergence_history = (
            self._build_global_convergence_history(
                cluster_results
            )
        )

        if len(
            convergence_history
        ) != self.iterations + 1:

            raise RuntimeError(
                "Invalid convergence history length. "
                f"Expected {self.iterations + 1}, "
                f"got {len(convergence_history)}."
            )

        # ============================================================
        # STEP 5: MERGE FINAL CLUSTER SOLUTIONS
        # ============================================================

        solution = (
            self._merge_solutions(
                cluster_results
            )
        )

        # ============================================================
        # STEP 6: VALIDATE MERGED SOLUTION
        # ============================================================

        if not self._validate_final_solution(
            solution
        ):
            raise RuntimeError(
                "Merged decomposed solution is invalid."
            )

        # ============================================================
        # STEP 7: EVALUATE BEFORE REPAIR
        # ============================================================

        pre_repair_result = (
            self._evaluate_solution(
                solution
            )
        )

        if not pre_repair_result.get(
            "valid",
            False,
        ):
            raise RuntimeError(
                "Merged solution failed VRP fitness validation."
            )

        pre_repair_fitness = (
            pre_repair_result["fitness"]
        )

        # ============================================================
        # DEFAULT REPAIR RESULT
        # ============================================================

        repair_result = {
            "solution": solution,

            "valid": pre_repair_result[
                "valid"
            ],

            "fitness": pre_repair_result[
                "fitness"
            ],

            "travel_time": pre_repair_result[
                "travel_time"
            ],

            "distance": pre_repair_result[
                "distance"
            ],

            "congestion": pre_repair_result[
                "congestion"
            ],

            "vehicle_results": pre_repair_result[
                "vehicle_results"
            ],

            "pre_repair_fitness": (
                pre_repair_fitness
            ),

            "repair_fitness": (
                pre_repair_fitness
            ),

            "improvement": 0.0,

            "repair_improvements": 0.0,

            "rounds": 0,

            "repair_rounds": 0,

            "operations": [],

            "repair_operations": 0,

            "evaluations": 0,
        }

        # ============================================================
        # STEP 8: FAST GLOBAL BOUNDARY REPAIR
        # ============================================================

        if self.repair:

            repairer = GlobalBoundaryRepair(
                problem=self.problem,
                fitness_function=self.fitness_function,
                max_rounds=self.repair_rounds,
                max_boundary_customers=(
                    self.max_boundary_customers
                ),
                max_target_vehicles=(
                    self.max_target_vehicles
                ),
                try_swaps=self.try_swaps,
            )

            repair_result = (
                repairer.repair(
                    solution
                )
            )

            solution = repair_result[
                "solution"
            ]

        # ============================================================
        # STEP 9: FINAL VALIDATION
        # ============================================================

        if not self._validate_final_solution(
            solution
        ):
            raise RuntimeError(
                "Final decomposed Warm-QPSO "
                "solution is invalid."
            )

        # ============================================================
        # STEP 10: FINAL EVALUATION
        # ============================================================

        final_result = (
            self._evaluate_solution(
                solution
            )
        )

        if not final_result.get(
            "valid",
            False,
        ):
            raise RuntimeError(
                "Final decomposed Warm-QPSO "
                "solution failed fitness validation."
            )

        # ============================================================
        # STEP 11: EXECUTION TIME
        # ============================================================

        execution_time = (
            time.perf_counter()
            - start_time
        )

        # ============================================================
        # STEP 12: BUILD CLUSTER SUMMARY
        # ============================================================

        cluster_summary = (
            self._build_cluster_summary(
                cluster_results
            )
        )

        # ============================================================
        # STEP 13: RETURN RESULT
        # ============================================================

        return {
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

            # --------------------------------------------------------
            # IMPORTANT:
            # This is now GLOBAL VRP fitness, not sum(cluster fitness).
            # --------------------------------------------------------

            "history": convergence_history,

            "iterations": self.iterations,

            "particles": self.particles,

            "beta": self.beta,

            "warm_particles": self.warm_particles,

            "warm_iterations": self.warm_iterations,

            "number_of_clusters": len(
                clusters
            ),

            "clusters": cluster_summary,

            # --------------------------------------------------------
            # Repair diagnostics.
            # --------------------------------------------------------

            "pre_repair_fitness": (
                repair_result.get(
                    "pre_repair_fitness",
                    pre_repair_fitness,
                )
            ),

            "repair_fitness": (
                repair_result.get(
                    "repair_fitness",
                    final_result["fitness"],
                )
            ),

            "repair_improvement": (
                repair_result.get(
                    "improvement",
                    0.0,
                )
            ),

            "repair_improvements": (
                repair_result.get(
                    "repair_improvements",
                    repair_result.get(
                        "improvements",
                        0,
                    ),
                )
            ),

            "repair_rounds": (
                repair_result.get(
                    "repair_rounds",
                    repair_result.get(
                        "rounds",
                        0,
                    ),
                )
            ),

            "repair_operations": (
                repair_result.get(
                    "repair_operations",
                    len(
                        repair_result.get(
                            "operations",
                            [],
                        )
                    ),
                )
            ),

            "repair_evaluations": (
                repair_result.get(
                    "evaluations",
                    0,
                )
            ),

            "operations": (
                repair_result.get(
                    "operations",
                    [],
                )
            ),

            "execution_time": execution_time,
        }
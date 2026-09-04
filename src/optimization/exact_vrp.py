"""
Exact Small-Instance VRP Solver
================================

This module provides an exact solver for SMALL capacitated
vehicle routing problems.

Purpose
-------
The solver is intended for benchmarking metaheuristic methods:

    PSO
    QPSO
    Warm-QPSO
    Decomposed Warm-QPSO

against a known optimal solution.

IMPORTANT
---------
This is an exhaustive exact solver and therefore should only
be used for small VRP instances.

It is not intended for large-scale optimization.
"""


import itertools
import time

from src.optimization.vrp_fitness import VRPFitnessFunction


class ExactVRPSolver:
    """
    Exhaustive exact solver for small Capacitated VRP instances.

    The solver enumerates customer assignments and route
    permutations and returns the globally best feasible solution
    according to the supplied VRP fitness function.
    """

    def __init__(
        self,
        problem,
        fitness_function=None,
        max_customers=9,
    ):
        self.problem = problem

        self.fitness_function = (
            fitness_function
            if fitness_function is not None
            else VRPFitnessFunction()
        )

        self.max_customers = int(
            max_customers
        )

        if self.max_customers < 1:
            raise ValueError(
                "max_customers must be at least 1."
            )

        if problem.number_of_customers > self.max_customers:
            raise ValueError(
                "ExactVRPSolver is intended only for "
                f"small instances with at most "
                f"{self.max_customers} customers. "
                f"Received "
                f"{problem.number_of_customers}."
            )

        if not problem.is_capacity_feasible():
            raise ValueError(
                "Vehicle capacity is insufficient "
                "for the selected problem."
            )

    # ========================================================
    # ROUTE DEMAND
    # ========================================================

    def _route_demand(self, route):
        """
        Calculate total demand of a vehicle route.
        """

        return sum(
            self.problem.get_customer(
                customer_id
            ).demand
            for customer_id in route
        )

    # ========================================================
    # VEHICLE CAPACITY
    # ========================================================

    def _vehicle_capacity(self, vehicle_id):
        """
        Return capacity of a vehicle.
        """

        return self.problem.get_vehicle(
            vehicle_id
        ).capacity

    # ========================================================
    # CAPACITY CHECK
    # ========================================================

    def _capacity_feasible(
        self,
        vehicle_id,
        route,
    ):
        """
        Check whether a route satisfies vehicle capacity.
        """

        demand = self._route_demand(
            route
        )

        capacity = self._vehicle_capacity(
            vehicle_id
        )

        return demand <= capacity

    # ========================================================
    # ROUTE EVALUATION
    # ========================================================

    def _evaluate_solution(self, solution):
        """
        Evaluate a complete VRP solution.
        """

        return self.fitness_function.evaluate(
            self.problem,
            solution,
        )

    # ========================================================
    # CUSTOMER ASSIGNMENTS
    # ========================================================

    def _generate_assignments(
        self,
        customer_ids,
        vehicle_ids,
    ):
        """
        Generate customer-to-vehicle assignments.

        Each customer is assigned exactly once.

        Example:

            customers = [C1, C2, C3]

            vehicles = [V1, V2]

        Possible assignment:

            V1: C1 C3
            V2: C2
        """

        number_of_customers = len(
            customer_ids
        )

        number_of_vehicles = len(
            vehicle_ids
        )

        for assignment in itertools.product(
            range(number_of_vehicles),
            repeat=number_of_customers,
        ):

            routes = {
                vehicle_id: []
                for vehicle_id in vehicle_ids
            }

            for index, vehicle_index in enumerate(
                assignment
            ):

                vehicle_id = vehicle_ids[
                    vehicle_index
                ]

                routes[
                    vehicle_id
                ].append(
                    customer_ids[index]
                )

            feasible = True

            for vehicle_id in vehicle_ids:

                if not self._capacity_feasible(
                    vehicle_id,
                    routes[vehicle_id],
                ):
                    feasible = False
                    break

            if feasible:
                yield routes

    # ========================================================
    # ROUTE PERMUTATIONS
    # ========================================================

    def _generate_route_permutations(
        self,
        assigned_customers,
    ):
        """
        Generate every possible customer ordering
        for a vehicle.
        """

        if not assigned_customers:
            yield []

            return

        yield from itertools.permutations(
            assigned_customers
        )

    # ========================================================
    # ROUTE COMBINATIONS
    # ========================================================

    def _generate_ordered_solutions(
        self,
        assignment,
        vehicle_ids,
    ):
        """
        Generate all ordered route solutions for a
        particular customer assignment.
        """

        route_options = []

        for vehicle_id in vehicle_ids:

            customers = assignment[
                vehicle_id
            ]

            permutations = list(
                self._generate_route_permutations(
                    customers
                )
            )

            route_options.append(
                permutations
            )

        for combination in itertools.product(
            *route_options
        ):

            solution = {}

            for index, vehicle_id in enumerate(
                vehicle_ids
            ):

                solution[
                    vehicle_id
                ] = list(
                    combination[index]
                )

            yield solution

    # ========================================================
    # SOLVE
    # ========================================================

    def solve(self):
        """
        Solve the VRP exactly.

        Returns
        -------
        dict

        Contains:

            valid
            solution
            fitness
            travel_time
            distance
            congestion
            vehicle_results
            execution_time
            evaluated_solutions
        """

        start_time = time.perf_counter()

        customer_ids = [
            customer.id
            for customer in self.problem.customers
        ]

        vehicle_ids = [
            vehicle.id
            for vehicle in self.problem.vehicles
        ]

        best_solution = None
        best_result = None

        evaluated_solutions = 0

        # ----------------------------------------------------
        # Enumerate assignments
        # ----------------------------------------------------

        for assignment in self._generate_assignments(
            customer_ids,
            vehicle_ids,
        ):

            # ------------------------------------------------
            # Enumerate route orderings
            # ------------------------------------------------

            for solution in self._generate_ordered_solutions(
                assignment,
                vehicle_ids,
            ):

                evaluated_solutions += 1

                try:
                    result = self._evaluate_solution(
                        solution
                    )

                except (ValueError, KeyError):
                    continue

                if not result.get(
                    "valid",
                    False,
                ):
                    continue

                if (
                    best_result is None
                    or result["fitness"]
                    < best_result["fitness"]
                ):

                    best_result = result

                    best_solution = {
                        vehicle_id: list(
                            solution[vehicle_id]
                        )
                        for vehicle_id in vehicle_ids
                    }

        # ----------------------------------------------------
        # No feasible solution
        # ----------------------------------------------------

        execution_time = (
            time.perf_counter()
            - start_time
        )

        if best_result is None:

            raise RuntimeError(
                "Exact solver could not find "
                "a feasible solution."
            )

        # ----------------------------------------------------
        # Return optimal result
        # ----------------------------------------------------

        return {
            "valid": True,

            "solution": best_solution,

            "fitness": best_result[
                "fitness"
            ],

            "travel_time": best_result[
                "travel_time"
            ],

            "distance": best_result[
                "distance"
            ],

            "congestion": best_result[
                "congestion"
            ],

            "vehicle_results": best_result.get(
                "vehicle_results",
                [],
            ),

            "execution_time": execution_time,

            "evaluated_solutions": (
                evaluated_solutions
            ),

            "optimal": True,

            "customers": (
                self.problem.number_of_customers
            ),

            "vehicles": (
                self.problem.number_of_vehicles
            ),
        }
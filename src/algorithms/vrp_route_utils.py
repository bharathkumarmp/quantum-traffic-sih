"""
VRP route utility functions.

Provides validation, capacity checking, customer assignment,
flattening, cloning, and greedy construction utilities.
"""

from typing import Dict, List, Optional


class RouteUtils:

    # ============================================================
    # EMPTY SOLUTION
    # ============================================================

    @staticmethod
    def empty_solution(problem) -> Dict[str, List[str]]:
        """Create an empty solution for all vehicles."""

        return {
            vehicle.id: []
            for vehicle in problem.vehicles
        }

    # ============================================================
    # CUSTOMER IDS
    # ============================================================

    @staticmethod
    def all_customer_ids(problem) -> List[str]:
        """Return all customer IDs."""

        return [
            customer.id
            for customer in problem.customers
        ]

    # ============================================================
    # DEMAND
    # ============================================================

    @staticmethod
    def total_route_demand(
        problem,
        customer_ids: List[str],
    ) -> float:
        """Calculate demand for a list of customers."""

        total = 0.0

        for customer_id in customer_ids:

            if customer_id not in problem.customer_map:
                raise ValueError(
                    f"Unknown customer: {customer_id}"
                )

            total += float(
                problem.customer_map[
                    customer_id
                ].demand
            )

        return total

    @staticmethod
    def vehicle_route_demand(
        problem,
        solution: Dict[str, List[str]],
        vehicle_id: str,
    ) -> float:
        """
        Calculate demand assigned to a vehicle.

        Expected call:

            vehicle_route_demand(
                problem,
                solution,
                "V1"
            )
        """

        vehicle_id = str(vehicle_id)

        if vehicle_id not in problem.vehicle_map:
            raise ValueError(
                f"Unknown vehicle: {vehicle_id}"
            )

        route = solution.get(
            vehicle_id,
            [],
        )

        return RouteUtils.total_route_demand(
            problem,
            route,
        )

    # ============================================================
    # CAPACITY
    # ============================================================

    @staticmethod
    def vehicle_capacity_feasible(
        problem,
        solution: Dict[str, List[str]],
        vehicle_id: str,
    ) -> bool:
        """Check whether one vehicle satisfies capacity."""

        vehicle_id = str(vehicle_id)

        if vehicle_id not in problem.vehicle_map:
            return False

        try:
            demand = (
                RouteUtils.vehicle_route_demand(
                    problem,
                    solution,
                    vehicle_id,
                )
            )
        except ValueError:
            return False

        capacity = float(
            problem.vehicle_map[
                vehicle_id
            ].capacity
        )

        return demand <= capacity

    @staticmethod
    def all_capacity_constraints_satisfied(
        problem,
        solution: Dict[str, List[str]],
    ) -> bool:
        """Check capacity constraints for every vehicle."""

        if not isinstance(solution, dict):
            return False

        for vehicle in problem.vehicles:

            if not RouteUtils.vehicle_capacity_feasible(
                problem,
                solution,
                vehicle.id,
            ):
                return False

        return True

    # ============================================================
    # CUSTOMER ASSIGNMENT
    # ============================================================

    @staticmethod
    def customer_assignment_valid(
        problem,
        solution: Dict[str, List[str]],
    ) -> bool:
        """
        Check that every customer is assigned exactly once.

        Returns False for:
        - duplicate customers
        - missing customers
        - unknown customers
        - invalid route structures
        """

        if not isinstance(solution, dict):
            return False

        expected_customers = {
            customer.id
            for customer in problem.customers
        }

        assigned_customers = []

        for route in solution.values():

            if not isinstance(route, list):
                return False

            for customer_id in route:

                if customer_id not in expected_customers:
                    return False

                assigned_customers.append(
                    customer_id
                )

        # Duplicate customer.
        if len(assigned_customers) != len(
            set(assigned_customers)
        ):
            return False

        # Missing customer.
        return (
            set(assigned_customers)
            == expected_customers
        )

    # ============================================================
    # SOLUTION STRUCTURE
    # ============================================================

    @staticmethod
    def solution_structure_valid(
        problem,
        solution: Dict[str, List[str]],
    ) -> bool:
        """Check vehicle keys and route types."""

        if not isinstance(solution, dict):
            return False

        expected_vehicle_ids = {
            vehicle.id
            for vehicle in problem.vehicles
        }

        if set(solution.keys()) != expected_vehicle_ids:
            return False

        return all(
            isinstance(route, list)
            for route in solution.values()
        )

    # ============================================================
    # VALID SOLUTION
    # ============================================================

    @staticmethod
    def is_valid_solution(
        problem,
        solution: Dict[str, List[str]],
    ) -> bool:
        """
        Check complete VRP solution validity.

        Requirements:
        1. Correct vehicle structure
        2. Every customer assigned exactly once
        3. All vehicle capacities respected
        """

        if not RouteUtils.solution_structure_valid(
            problem,
            solution,
        ):
            return False

        if not RouteUtils.customer_assignment_valid(
            problem,
            solution,
        ):
            return False

        if not RouteUtils.all_capacity_constraints_satisfied(
            problem,
            solution,
        ):
            return False

        return True

    # ============================================================
    # GREEDY ASSIGNMENT
    # ============================================================

    @staticmethod
    def assign_customers_greedily(
        problem,
        customer_ids: Optional[List[str]] = None,
    ):
        """
        Greedily assign customers to vehicles.

        Returns None if no feasible assignment can be
        constructed.
        """

        if customer_ids is None:
            customer_ids = (
                RouteUtils.all_customer_ids(
                    problem
                )
            )

        solution = (
            RouteUtils.empty_solution(
                problem
            )
        )

        remaining_capacity = {
            vehicle.id: float(
                vehicle.capacity
            )
            for vehicle in problem.vehicles
        }

        for customer_id in customer_ids:

            if customer_id not in problem.customer_map:
                return None

            demand = float(
                problem.customer_map[
                    customer_id
                ].demand
            )

            assigned = False

            for vehicle in problem.vehicles:

                vehicle_id = vehicle.id

                if (
                    remaining_capacity[
                        vehicle_id
                    ] >= demand
                ):

                    solution[
                        vehicle_id
                    ].append(
                        customer_id
                    )

                    remaining_capacity[
                        vehicle_id
                    ] -= demand

                    assigned = True
                    break

            if not assigned:
                return None

        if not RouteUtils.is_valid_solution(
            problem,
            solution,
        ):
            return None

        return solution

    # ============================================================
    # FLATTEN SOLUTION
    # ============================================================

    @staticmethod
    def flatten_solution(
        problem,
        solution: Dict[str, List[str]],
    ):
        """
        Flatten a solution into:

            [
                ("V1", "C1"),
                ("V1", "C3"),
                ("V2", "C2"),
                ...
            ]
        """

        result = []

        for vehicle in problem.vehicles:

            vehicle_id = vehicle.id

            route = solution.get(
                vehicle_id,
                [],
            )

            for customer_id in route:

                result.append(
                    (
                        vehicle_id,
                        customer_id,
                    )
                )

        return result

    # ============================================================
    # CLONE SOLUTION
    # ============================================================

    @staticmethod
    def clone_solution(
        solution: Dict[str, List[str]],
    ):
        """Create a safe copy of a solution."""

        if solution is None:
            return None

        return {
            vehicle_id: list(customer_ids)
            for vehicle_id, customer_ids
            in solution.items()
        }


# =================================================================
# MODULE-LEVEL COMPATIBILITY FUNCTIONS
# =================================================================

def empty_solution(problem):
    return RouteUtils.empty_solution(problem)


def all_customer_ids(problem):
    return RouteUtils.all_customer_ids(problem)


def total_route_demand(
    problem,
    customer_ids,
):
    return RouteUtils.total_route_demand(
        problem,
        customer_ids,
    )


def vehicle_route_demand(
    problem,
    solution,
    vehicle_id,
):
    return RouteUtils.vehicle_route_demand(
        problem,
        solution,
        vehicle_id,
    )


def vehicle_capacity_feasible(
    problem,
    solution,
    vehicle_id,
):
    return RouteUtils.vehicle_capacity_feasible(
        problem,
        solution,
        vehicle_id,
    )


def all_capacity_constraints_satisfied(
    problem,
    solution,
):
    return RouteUtils.all_capacity_constraints_satisfied(
        problem,
        solution,
    )


def customer_assignment_valid(
    problem,
    solution,
):
    return RouteUtils.customer_assignment_valid(
        problem,
        solution,
    )


def solution_structure_valid(
    problem,
    solution,
):
    return RouteUtils.solution_structure_valid(
        problem,
        solution,
    )


def is_valid_solution(
    problem,
    solution,
):
    return RouteUtils.is_valid_solution(
        problem,
        solution,
    )


def assign_customers_greedily(
    problem,
    customer_ids=None,
):
    return RouteUtils.assign_customers_greedily(
        problem,
        customer_ids,
    )


def flatten_solution(
    problem,
    solution,
):
    return RouteUtils.flatten_solution(
        problem,
        solution,
    )


def clone_solution(solution):
    return RouteUtils.clone_solution(
        solution
    )


# =================================================================
# BACKWARD-COMPATIBILITY ALIAS
# =================================================================

VRPRouteUtils = RouteUtils
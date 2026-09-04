"""
Fast Global Boundary Repair for Decomposed VRP Solutions.

This module performs a lightweight greedy local-search repair after
cluster-level optimization.

Pipeline:

    Decomposed Warm-QPSO
            |
            v
       Merge solutions
            |
            v
    Boundary customer search
            |
            v
      Move / Swap search
            |
            v
      Accept improvement
            |
            v
        Final solution

Unlike exhaustive repair, this implementation limits the search space
and is intended to remain practical for larger VRP instances.
"""

from copy import deepcopy


class GlobalBoundaryRepair:
    """
    Fast greedy local-search repair for a merged VRP solution.

    Parameters
    ----------
    problem:
        VRPProblem instance.

    fitness_function:
        VRPFitnessFunction instance.

    max_rounds:
        Maximum number of improvement rounds.

    max_boundary_customers:
        Maximum number of customers considered during each repair round.

    max_target_vehicles:
        Maximum number of target vehicles considered for a move.

    try_swaps:
        Whether to try exchanging two customers between vehicles.
    """

    def __init__(
        self,
        problem,
        fitness_function,
        max_rounds=2,
        max_boundary_customers=20,
        max_target_vehicles=5,
        try_swaps=True,
    ):
        if problem is None:
            raise ValueError("problem is required.")

        if fitness_function is None:
            raise ValueError("fitness_function is required.")

        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")

        if max_boundary_customers < 1:
            raise ValueError(
                "max_boundary_customers must be at least 1."
            )

        if max_target_vehicles < 1:
            raise ValueError(
                "max_target_vehicles must be at least 1."
            )

        self.problem = problem
        self.fitness_function = fitness_function

        self.max_rounds = int(max_rounds)
        self.max_boundary_customers = int(
            max_boundary_customers
        )
        self.max_target_vehicles = int(
            max_target_vehicles
        )
        self.try_swaps = bool(try_swaps)

        # Stores previously evaluated solutions.
        self._evaluation_cache = {}

    # ============================================================
    # SOLUTION VALIDATION
    # ============================================================

    def _validate_solution_structure(self, solution):
        """
        Validate the basic VRP solution structure.

        A valid solution must:

        1. Be a dictionary.
        2. Contain exactly all problem vehicles.
        3. Contain only valid customers.
        4. Assign every customer exactly once.
        """

        if not isinstance(solution, dict):
            return False

        expected_vehicle_ids = {
            str(vehicle.id)
            for vehicle in self.problem.vehicles
        }

        actual_vehicle_ids = {
            str(vehicle_id)
            for vehicle_id in solution.keys()
        }

        # Exact vehicle-set validation.
        if actual_vehicle_ids != expected_vehicle_ids:
            return False

        expected_customer_ids = {
            str(customer.id)
            for customer in self.problem.customers
        }

        seen_customers = set()

        for vehicle_id, route in solution.items():

            if not isinstance(route, list):
                return False

            for customer_id in route:

                customer_id = str(customer_id)

                # Unknown customer.
                if customer_id not in expected_customer_ids:
                    return False

                # Duplicate customer.
                if customer_id in seen_customers:
                    return False

                seen_customers.add(customer_id)

        # Every customer must be assigned.
        return seen_customers == expected_customer_ids

    # ============================================================
    # HASHABLE SOLUTION KEY
    # ============================================================

    def _solution_key(self, solution):
        """
        Convert a solution into a hashable representation.

        Used for fitness-evaluation caching.
        """

        return tuple(
            (
                str(vehicle_id),
                tuple(
                    str(customer_id)
                    for customer_id in route
                ),
            )
            for vehicle_id, route in sorted(
                solution.items(),
                key=lambda item: str(item[0]),
            )
        )

    # ============================================================
    # FITNESS EVALUATION
    # ============================================================

    def _evaluate(self, solution):
        """
        Evaluate a solution.

        Results are cached to avoid recalculating the same solution.
        """

        key = self._solution_key(solution)

        if key not in self._evaluation_cache:

            result = self.fitness_function.evaluate(
                problem=self.problem,
                solution=solution,
            )

            self._evaluation_cache[key] = result

        return self._evaluation_cache[key]

    # ============================================================
    # ROUTE DEMAND
    # ============================================================

    def _route_demand(self, route):
        """
        Calculate total demand of a vehicle route.
        """

        total_demand = 0.0

        for customer_id in route:

            customer = self.problem.get_customer(
                customer_id
            )

            total_demand += customer.demand

        return total_demand

    # ============================================================
    # VEHICLE CAPACITY
    # ============================================================

    def _vehicle_capacity(self, vehicle_id):
        """
        Return vehicle capacity.
        """

        vehicle = self.problem.get_vehicle(
            vehicle_id
        )

        return vehicle.capacity

    # ============================================================
    # CAPACITY CHECK
    # ============================================================

    def _can_accept_customer(
        self,
        solution,
        target_vehicle_id,
        customer_id,
    ):
        """
        Check whether target vehicle can accept a customer.
        """

        target_route = solution[target_vehicle_id]

        current_demand = self._route_demand(
            target_route
        )

        customer = self.problem.get_customer(
            customer_id
        )

        new_demand = (
            current_demand
            + customer.demand
        )

        capacity = self._vehicle_capacity(
            target_vehicle_id
        )

        return new_demand <= capacity + 1e-9

    # ============================================================
    # BOUNDARY CUSTOMER SELECTION
    # ============================================================

    def _boundary_customers(self, solution):
        """
        Select a limited set of promising customers.

        The first candidates are the beginning and ending customers
        of each vehicle route.

        If more candidates are required, additional customers are
        taken from the routes.

        This prevents the repair algorithm from examining every
        possible customer.
        """

        candidates = []

        # --------------------------------------------------------
        # First pass: route boundaries
        # --------------------------------------------------------

        for vehicle_id in sorted(
            solution.keys(),
            key=str,
        ):

            route = solution[vehicle_id]

            if not route:
                continue

            # First customer.
            candidates.append(
                (
                    str(vehicle_id),
                    str(route[0]),
                )
            )

            # Last customer.
            if len(route) > 1:

                candidates.append(
                    (
                        str(vehicle_id),
                        str(route[-1]),
                    )
                )

        # --------------------------------------------------------
        # Remove duplicate boundary customers.
        # --------------------------------------------------------

        unique = []
        seen = set()

        for vehicle_id, customer_id in candidates:

            key = (
                vehicle_id,
                customer_id,
            )

            if key not in seen:

                seen.add(key)
                unique.append(key)

        # --------------------------------------------------------
        # Add additional customers if necessary.
        # --------------------------------------------------------

        if len(unique) < self.max_boundary_customers:

            for vehicle_id in sorted(
                solution.keys(),
                key=str,
            ):

                route = solution[vehicle_id]

                for customer_id in route:

                    key = (
                        str(vehicle_id),
                        str(customer_id),
                    )

                    if key in seen:
                        continue

                    seen.add(key)
                    unique.append(key)

                    if (
                        len(unique)
                        >= self.max_boundary_customers
                    ):
                        break

                if (
                    len(unique)
                    >= self.max_boundary_customers
                ):
                    break

        return unique[
            :self.max_boundary_customers
        ]

    # ============================================================
    # MOVE CUSTOMER
    # ============================================================

    def _move_customer(
        self,
        solution,
        source_vehicle,
        target_vehicle,
        customer_id,
        target_position=None,
    ):
        """
        Create a candidate solution by moving one customer
        from one vehicle route to another.
        """

        candidate = deepcopy(solution)

        source_route = candidate[
            source_vehicle
        ]

        target_route = candidate[
            target_vehicle
        ]

        customer_id = str(customer_id)

        if customer_id not in source_route:
            return None

        source_route.remove(customer_id)

        # --------------------------------------------------------
        # Append customer.
        # --------------------------------------------------------

        if target_position is None:

            target_route.append(
                customer_id
            )

        # --------------------------------------------------------
        # Insert customer at a specific position.
        # --------------------------------------------------------

        else:

            target_position = max(
                0,
                min(
                    int(target_position),
                    len(target_route),
                ),
            )

            target_route.insert(
                target_position,
                customer_id,
            )

        return candidate

    # ============================================================
    # TARGET VEHICLES
    # ============================================================

    def _candidate_target_vehicles(
        self,
        solution,
        source_vehicle,
    ):
        """
        Return a limited number of target vehicles.

        Vehicles with available capacity are preferred.
        """

        targets = []

        for vehicle_id in sorted(
            solution.keys(),
            key=str,
        ):

            vehicle_id = str(vehicle_id)

            if vehicle_id == str(source_vehicle):
                continue

            targets.append(vehicle_id)

        return targets[
            :self.max_target_vehicles
        ]

    # ============================================================
    # BEST MOVE
    # ============================================================

    def _find_best_move(
        self,
        solution,
        current_fitness,
    ):
        """
        Find the best improving single-customer move.

        Only selected boundary customers are considered.
        """

        best_solution = None
        best_fitness = current_fitness
        best_operation = None

        boundary = self._boundary_customers(
            solution
        )

        for (
            source_vehicle,
            customer_id,
        ) in boundary:

            source_route = solution[
                source_vehicle
            ]

            if customer_id not in source_route:
                continue

            target_vehicles = (
                self._candidate_target_vehicles(
                    solution,
                    source_vehicle,
                )
            )

            for target_vehicle in target_vehicles:

                # ------------------------------------------------
                # Capacity pruning.
                # ------------------------------------------------

                if not self._can_accept_customer(
                    solution,
                    target_vehicle,
                    customer_id,
                ):
                    continue

                target_route = solution[
                    target_vehicle
                ]

                # ------------------------------------------------
                # Fast strategy:
                # only try appending the customer.
                #
                # This avoids testing every insertion position.
                # ------------------------------------------------

                candidate = self._move_customer(
                    solution,
                    source_vehicle,
                    target_vehicle,
                    customer_id,
                    target_position=len(
                        target_route
                    ),
                )

                if candidate is None:
                    continue

                result = self._evaluate(
                    candidate
                )

                candidate_fitness = (
                    result["fitness"]
                )

                # ------------------------------------------------
                # Accept only strict improvements.
                # ------------------------------------------------

                if (
                    candidate_fitness
                    < best_fitness - 1e-9
                ):

                    best_fitness = (
                        candidate_fitness
                    )

                    best_solution = candidate

                    best_operation = {
                        "type": "move",
                        "customer": customer_id,
                        "from": source_vehicle,
                        "to": target_vehicle,
                    }

        return (
            best_solution,
            best_fitness,
            best_operation,
        )

    # ============================================================
    # BEST SWAP
    # ============================================================

    def _find_best_swap(
        self,
        solution,
        current_fitness,
    ):
        """
        Find the best improving swap between two vehicles.

        Only selected boundary customers are considered.
        """

        if not self.try_swaps:

            return (
                None,
                current_fitness,
                None,
            )

        boundary = self._boundary_customers(
            solution
        )

        best_solution = None
        best_fitness = current_fitness
        best_operation = None

        # --------------------------------------------------------
        # Compare pairs of boundary customers.
        # --------------------------------------------------------

        for i in range(len(boundary)):

            (
                vehicle_a,
                customer_a,
            ) = boundary[i]

            for j in range(i + 1, len(boundary)):

                (
                    vehicle_b,
                    customer_b,
                ) = boundary[j]

                # Customers from same vehicle cannot be swapped.
                if vehicle_a == vehicle_b:
                    continue

                route_a = solution[
                    vehicle_a
                ]

                route_b = solution[
                    vehicle_b
                ]

                if customer_a not in route_a:
                    continue

                if customer_b not in route_b:
                    continue

                # ------------------------------------------------
                # Calculate current demands.
                # ------------------------------------------------

                demand_a = self._route_demand(
                    route_a
                )

                demand_b = self._route_demand(
                    route_b
                )

                customer_a_obj = (
                    self.problem.get_customer(
                        customer_a
                    )
                )

                customer_b_obj = (
                    self.problem.get_customer(
                        customer_b
                    )
                )

                # ------------------------------------------------
                # Calculate demands after swap.
                # ------------------------------------------------

                new_demand_a = (
                    demand_a
                    - customer_a_obj.demand
                    + customer_b_obj.demand
                )

                new_demand_b = (
                    demand_b
                    - customer_b_obj.demand
                    + customer_a_obj.demand
                )

                capacity_a = (
                    self._vehicle_capacity(
                        vehicle_a
                    )
                )

                capacity_b = (
                    self._vehicle_capacity(
                        vehicle_b
                    )
                )

                # ------------------------------------------------
                # Capacity pruning.
                # ------------------------------------------------

                if (
                    new_demand_a
                    > capacity_a + 1e-9
                ):
                    continue

                if (
                    new_demand_b
                    > capacity_b + 1e-9
                ):
                    continue

                # ------------------------------------------------
                # Build candidate.
                # ------------------------------------------------

                candidate = deepcopy(
                    solution
                )

                index_a = (
                    candidate[vehicle_a].index(
                        customer_a
                    )
                )

                index_b = (
                    candidate[vehicle_b].index(
                        customer_b
                    )
                )

                candidate[vehicle_a][
                    index_a
                ] = customer_b

                candidate[vehicle_b][
                    index_b
                ] = customer_a

                # ------------------------------------------------
                # Evaluate.
                # ------------------------------------------------

                result = self._evaluate(
                    candidate
                )

                candidate_fitness = (
                    result["fitness"]
                )

                # ------------------------------------------------
                # Accept strict improvement.
                # ------------------------------------------------

                if (
                    candidate_fitness
                    < best_fitness - 1e-9
                ):

                    best_fitness = (
                        candidate_fitness
                    )

                    best_solution = candidate

                    best_operation = {
                        "type": "swap",
                        "customer_a": customer_a,
                        "vehicle_a": vehicle_a,
                        "customer_b": customer_b,
                        "vehicle_b": vehicle_b,
                    }

        return (
            best_solution,
            best_fitness,
            best_operation,
        )

    # ============================================================
    # ONE IMPROVEMENT STEP
    # ============================================================

    def _improve_once(
        self,
        solution,
        current_fitness,
    ):
        """
        Perform one greedy improvement step.

        Both a move and swap search are performed.

        The better improvement is selected.
        """

        (
            move_solution,
            move_fitness,
            move_operation,
        ) = self._find_best_move(
            solution,
            current_fitness,
        )

        (
            swap_solution,
            swap_fitness,
            swap_operation,
        ) = self._find_best_swap(
            solution,
            current_fitness,
        )

        # --------------------------------------------------------
        # No improvement.
        # --------------------------------------------------------

        if (
            move_solution is None
            and swap_solution is None
        ):

            return (
                solution,
                current_fitness,
                None,
            )

        # --------------------------------------------------------
        # Move is better.
        # --------------------------------------------------------

        if (
            move_solution is not None
            and (
                swap_solution is None
                or move_fitness <= swap_fitness
            )
        ):

            return (
                move_solution,
                move_fitness,
                move_operation,
            )

        # --------------------------------------------------------
        # Swap is better.
        # --------------------------------------------------------

        return (
            swap_solution,
            swap_fitness,
            swap_operation,
        )

    # ============================================================
    # MAIN REPAIR FUNCTION
    # ============================================================

    def repair(self, solution):
        """
        Perform fast global boundary repair.

        Parameters
        ----------
        solution:
            Merged solution produced by the decomposition stage.

        Returns
        -------
        dict
            Repaired solution and diagnostic information.
        """

        # --------------------------------------------------------
        # Validate input.
        # --------------------------------------------------------

        if not self._validate_solution_structure(
            solution
        ):
            raise ValueError(
                "Invalid VRP solution supplied "
                "to GlobalBoundaryRepair."
            )

        # --------------------------------------------------------
        # Work on a copy.
        # --------------------------------------------------------

        current_solution = deepcopy(
            solution
        )

        # --------------------------------------------------------
        # Reset evaluation cache.
        # --------------------------------------------------------

        self._evaluation_cache.clear()

        # --------------------------------------------------------
        # Evaluate initial solution.
        # --------------------------------------------------------

        initial_result = self._evaluate(
            current_solution
        )

        initial_fitness = (
            initial_result["fitness"]
        )

        current_fitness = initial_fitness

        # --------------------------------------------------------
        # Diagnostics.
        # --------------------------------------------------------

        operations = []

        rounds_completed = 0

        no_improvement_rounds = 0

        # --------------------------------------------------------
        # Greedy local search.
        # --------------------------------------------------------

        for round_number in range(
            1,
            self.max_rounds + 1,
        ):

            fitness_before = (
                current_fitness
            )

            (
                improved_solution,
                improved_fitness,
                operation,
            ) = self._improve_once(
                current_solution,
                current_fitness,
            )

            rounds_completed = round_number

            # ----------------------------------------------------
            # No improvement.
            # ----------------------------------------------------

            if operation is None:

                no_improvement_rounds += 1

                break

            # ----------------------------------------------------
            # Accept improvement.
            # ----------------------------------------------------

            current_solution = (
                improved_solution
            )

            current_fitness = (
                improved_fitness
            )

            operations.append(
                {
                    "round": round_number,
                    "operation": operation,
                    "fitness_before": fitness_before,
                    "fitness_after": improved_fitness,
                    "improvement": (
                        fitness_before
                        - improved_fitness
                    ),
                }
            )

        # --------------------------------------------------------
        # Final evaluation.
        # --------------------------------------------------------

        final_result = self._evaluate(
            current_solution
        )

        final_fitness = (
            final_result["fitness"]
        )

        total_improvement = (
            initial_fitness
            - final_fitness
        )

        # --------------------------------------------------------
        # Return complete result.
        # --------------------------------------------------------

        return {
            "solution": current_solution,

            "fitness": final_fitness,

            "travel_time": (
                final_result["travel_time"]
            ),

            "distance": (
                final_result["distance"]
            ),

            "congestion": (
                final_result["congestion"]
            ),

            "vehicle_results": (
                final_result["vehicle_results"]
            ),

            # Before/after repair.
            "pre_repair_fitness": (
                initial_fitness
            ),

            "repair_fitness": (
                final_fitness
            ),

            "improvement": (
                total_improvement
            ),

            "repair_improvements": (
                total_improvement
            ),

            # Diagnostics.
            "rounds": rounds_completed,

            "repair_rounds": (
                rounds_completed
            ),

            "operations": operations,

            "repair_operations": len(
                operations
            ),

            "no_improvement_rounds": (
                no_improvement_rounds
            ),

            # Number of unique fitness evaluations.
            "evaluations": len(
                self._evaluation_cache
            ),
        }
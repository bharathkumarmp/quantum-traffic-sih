class VRPProblem:
    """
    Vehicle Routing Problem definition.

    Contains:

        graph
        depot
        customers
        vehicles

    and provides basic feasibility information.
    """

    def __init__(
        self,
        graph,
        depot,
        customers,
        vehicles,
    ):
        if not customers:
            raise ValueError(
                "At least one customer is required."
            )

        if not vehicles:
            raise ValueError(
                "At least one vehicle is required."
            )

        self.graph = graph
        self.depot = str(depot)

        self.customers = list(customers)
        self.vehicles = list(vehicles)

        self.customer_map = {
            customer.id: customer
            for customer in self.customers
        }

        self.vehicle_map = {
            vehicle.id: vehicle
            for vehicle in self.vehicles
        }

    @property
    def number_of_customers(self):
        return len(self.customers)

    @property
    def number_of_vehicles(self):
        return len(self.vehicles)

    @property
    def total_demand(self):
        return sum(
            customer.demand
            for customer in self.customers
        )

    @property
    def total_capacity(self):
        return sum(
            vehicle.capacity
            for vehicle in self.vehicles
        )

    def is_capacity_feasible(self):
        return (
            self.total_demand
            <= self.total_capacity
        )

    def get_customer(self, customer_id):
        return self.customer_map[
            str(customer_id)
        ]

    def get_vehicle(self, vehicle_id):
        return self.vehicle_map[
            str(vehicle_id)
        ]

    def __repr__(self):
        return (
            f"VRPProblem("
            f"customers={self.number_of_customers}, "
            f"vehicles={self.number_of_vehicles}, "
            f"total_demand={self.total_demand}, "
            f"total_capacity={self.total_capacity}"
            f")"
        )
from src.domain.customer import Customer
from src.domain.vehicle import Vehicle
from src.domain.vrp_problem import VRPProblem


def test_vrp_foundation():
    # Customers
    customers = [
        Customer("C1", 4, "N1"),
        Customer("C2", 7, "N2"),
        Customer("C3", 3, "N3"),
        Customer("C4", 5, "N4"),
        Customer("C5", 6, "N5"),
        Customer("C6", 2, "N6"),
        Customer("C7", 4, "N7"),
        Customer("C8", 3, "N8"),
        Customer("C9", 5, "N9"),
        Customer("C10", 2, "N10"),
    ]

    # Vehicles
    vehicles = [
        Vehicle("V1", 25, "N0", "N0"),
        Vehicle("V2", 25, "N0", "N0"),
    ]

    # Graph is not required yet for this foundation test
    problem = VRPProblem(
        graph=None,
        depot="N0",
        customers=customers,
        vehicles=vehicles,
    )

    print("\n===== VRP FOUNDATION =====")
    print("Customers:", problem.number_of_customers)
    print("Vehicles:", problem.number_of_vehicles)
    print("Total demand:", problem.total_demand)
    print("Total capacity:", problem.total_capacity)
    print("Capacity feasible:", problem.is_capacity_feasible())

    assert problem.number_of_customers == 10
    assert problem.number_of_vehicles == 2
    assert problem.total_demand == 41
    assert problem.total_capacity == 50
    assert problem.is_capacity_feasible()


if __name__ == "__main__":
    test_vrp_foundation()
from dataclasses import dataclass


@dataclass
class Customer:
    """
    Represents a customer that must be visited
    by a vehicle.
    """

    id: str
    demand: float
    node_id: str

    def __post_init__(self):
        self.id = str(self.id)
        self.node_id = str(self.node_id)
        self.demand = float(self.demand)

        if self.demand < 0:
            raise ValueError(
                "Customer demand cannot be negative."
            )
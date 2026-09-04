from dataclasses import dataclass


@dataclass
class Vehicle:
    """
    Represents a delivery vehicle.
    """

    id: str
    capacity: float
    start_node: str
    end_node: str

    def __post_init__(self):
        self.id = str(self.id)
        self.start_node = str(self.start_node)
        self.end_node = str(self.end_node)

        self.capacity = float(self.capacity)

        if self.capacity <= 0:
            raise ValueError(
                "Vehicle capacity must be positive."
            )
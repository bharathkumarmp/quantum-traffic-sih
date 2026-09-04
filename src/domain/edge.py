from dataclasses import dataclass


@dataclass
class Edge:
    source: object
    target: object
    distance: float
    travel_time: float
    congestion: float = 0.0
    blocked: bool = False

    def __post_init__(self):
        self.distance = float(self.distance)
        self.travel_time = float(self.travel_time)
        self.congestion = float(self.congestion)
        self.blocked = bool(self.blocked)

        if self.distance < 0:
            raise ValueError("Distance cannot be negative.")

        if self.travel_time < 0:
            raise ValueError("Travel time cannot be negative.")

        if self.congestion < 0:
            raise ValueError("Congestion cannot be negative.")
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """
    Central configuration for the Q-ROUTE optimization engine.

    This object keeps all optimization parameters in one place so that
    the dashboard, experiments, and algorithms can use the same settings.
    """

    # ---------------------------------------------------------
    # General optimization parameters
    # ---------------------------------------------------------

    particles: int = 20
    iterations: int = 50
    seed: int = 42

    # ---------------------------------------------------------
    # PSO parameters
    # ---------------------------------------------------------

    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5

    # ---------------------------------------------------------
    # QPSO parameters
    # ---------------------------------------------------------

    beta: float = 0.7

    # ---------------------------------------------------------
    # Warm-QPSO parameters
    # ---------------------------------------------------------

    warm_iterations: int = 10
    warm_particles: int = 10

    # ---------------------------------------------------------
    # Decomposition parameters
    # ---------------------------------------------------------

    clusters: int = 5

    # ---------------------------------------------------------
    # Global boundary repair
    # ---------------------------------------------------------

    repair_enabled: bool = True
    repair_rounds: int = 2
    max_boundary_customers: int = 20
    max_target_vehicles: int = 5
    try_swaps: bool = True

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def __post_init__(self):
        self.particles = int(self.particles)
        self.iterations = int(self.iterations)
        self.seed = int(self.seed)

        self.inertia = float(self.inertia)
        self.cognitive = float(self.cognitive)
        self.social = float(self.social)

        self.beta = float(self.beta)

        self.warm_iterations = int(self.warm_iterations)
        self.warm_particles = int(self.warm_particles)

        self.clusters = int(self.clusters)

        self.repair_rounds = int(self.repair_rounds)
        self.max_boundary_customers = int(
            self.max_boundary_customers
        )
        self.max_target_vehicles = int(
            self.max_target_vehicles
        )

        if self.particles < 1:
            raise ValueError("particles must be at least 1.")

        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")

        if self.inertia < 0:
            raise ValueError("inertia cannot be negative.")

        if self.cognitive < 0:
            raise ValueError("cognitive cannot be negative.")

        if self.social < 0:
            raise ValueError("social cannot be negative.")

        if self.beta <= 0:
            raise ValueError("beta must be positive.")

        if self.warm_iterations < 1:
            raise ValueError(
                "warm_iterations must be at least 1."
            )

        if self.warm_particles < 2:
            raise ValueError(
                "warm_particles must be at least 2."
            )

        if self.clusters < 1:
            raise ValueError(
                "clusters must be at least 1."
            )

        if self.repair_rounds < 0:
            raise ValueError(
                "repair_rounds cannot be negative."
            )

        if self.max_boundary_customers < 1:
            raise ValueError(
                "max_boundary_customers must be at least 1."
            )

        if self.max_target_vehicles < 1:
            raise ValueError(
                "max_target_vehicles must be at least 1."
            )

    def to_dict(self):
        """
        Return configuration as a dictionary.
        """

        return {
            "particles": self.particles,
            "iterations": self.iterations,
            "seed": self.seed,
            "inertia": self.inertia,
            "cognitive": self.cognitive,
            "social": self.social,
            "beta": self.beta,
            "warm_iterations": self.warm_iterations,
            "warm_particles": self.warm_particles,
            "clusters": self.clusters,
            "repair_enabled": self.repair_enabled,
            "repair_rounds": self.repair_rounds,
            "max_boundary_customers": self.max_boundary_customers,
            "max_target_vehicles": self.max_target_vehicles,
            "try_swaps": self.try_swaps,
        } 
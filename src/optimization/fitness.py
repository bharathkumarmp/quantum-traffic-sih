class FitnessFunction:
    """
    Multi-objective traffic-route fitness.

    Lower fitness is better.

    F(R) =
        wt * normalized_travel_time
      + wd * normalized_distance
      + wc * normalized_congestion
      + penalty
    """

    def __init__(
        self,
        time_weight=0.5,
        distance_weight=0.3,
        congestion_weight=0.2,
        invalid_penalty=1_000_000.0,
    ):
        total = (
            time_weight
            + distance_weight
            + congestion_weight
        )

        if total <= 0:
            raise ValueError(
                "Fitness weights must have positive sum."
            )

        self.time_weight = (
            time_weight / total
        )

        self.distance_weight = (
            distance_weight / total
        )

        self.congestion_weight = (
            congestion_weight / total
        )

        self.invalid_penalty = (
            float(invalid_penalty)
        )

    def calculate(
        self,
        travel_time,
        distance,
        congestion,
        penalty=0.0,
    ):
        """
        Calculate route fitness.

        Parameters
        ----------
        travel_time : float
            Total route travel time.

        distance : float
            Total route distance.

        congestion : float
            Average route congestion.

        penalty : float
            Constraint violation penalty.
        """

        if travel_time < 0:
            raise ValueError(
                "Travel time cannot be negative."
            )

        if distance < 0:
            raise ValueError(
                "Distance cannot be negative."
            )

        if congestion < 0:
            raise ValueError(
                "Congestion cannot be negative."
            )

        # For the initial implementation,
        # traffic values are normalized using
        # the expected ranges:
        #
        # congestion: [0, 1]
        #
        # Time and distance normalization will
        # be handled at scenario level later.

        normalized_congestion = min(
            1.0,
            congestion,
        )

        cost = (
            self.congestion_weight
            * normalized_congestion
        )

        # Time and distance are intentionally
        # passed through here. Scenario-level
        # normalization will be added before
        # large benchmark experiments.

        cost += (
            self.time_weight
            * travel_time
        )

        cost += (
            self.distance_weight
            * distance
        )

        cost += float(penalty)

        return cost

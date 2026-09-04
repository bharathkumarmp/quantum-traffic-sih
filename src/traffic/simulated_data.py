import csv
from pathlib import Path


class SimulatedTrafficData:

    def __init__(self, csv_path=None):

        if csv_path is None:
            project_root = Path(__file__).resolve().parents[2]

            csv_path = (
                project_root
                / "data"
                / "scenarios"
                / "simulated_traffic.csv"
            )

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Traffic dataset not found: {self.csv_path}"
            )

        self.records = self._load()

    def _load(self):

        with open(
            self.csv_path,
            "r",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            records = []

            for row in reader:

                records.append({
                    "timestamp": row["timestamp"],
                    "scenario": row["scenario"],
                    "edge_id": row["edge_id"],
                    "source": row["source"],
                    "target": row["target"],
                    "distance_km": float(
                        row["distance_km"]
                    ),
                    "free_speed_kmph": float(
                        row["free_speed_kmph"]
                    ),
                    "traffic_speed_kmph": float(
                        row["traffic_speed_kmph"]
                    ),
                    "congestion": float(
                        row["congestion"]
                    ),
                    "blocked": (
                        row["blocked"].lower()
                        == "true"
                    ),
                })

            return records

    def scenarios(self):

        return sorted(
            set(
                record["scenario"]
                for record in self.records
            )
        )

    def timestamps(self):

        return sorted(
            set(
                record["timestamp"]
                for record in self.records
            )
        )

    def get_scenario(self, scenario):

        return [
            record
            for record in self.records
            if record["scenario"].lower()
            == scenario.lower()
        ]

    def get_timestamp(self, timestamp):

        return [
            record
            for record in self.records
            if record["timestamp"]
            == timestamp
        ]

    def get(self, scenario, timestamp=None):

        records = self.get_scenario(scenario)

        if timestamp is not None:

            records = [
                record
                for record in records
                if record["timestamp"]
                == timestamp
            ]

        return records

    def summary(self, scenario):

        records = self.get_scenario(scenario)

        if not records:
            raise ValueError(
                f"Unknown traffic scenario: {scenario}"
            )

        congestion = sum(
            r["congestion"]
            for r in records
        ) / len(records)

        blocked = sum(
            1
            for r in records
            if r["blocked"]
        )

        return {
            "scenario": scenario,
            "records": len(records),
            "average_congestion": congestion,
            "blocked_edges": blocked,
        }
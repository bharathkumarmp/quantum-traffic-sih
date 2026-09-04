from src.data.scenario_graph_generator import ScenarioGraphGenerator
from src.algorithms.dijkstra import Dijkstra
from src.algorithms.pso import PSO
from src.algorithms.qpso import QPSO


def route_ids(result):
    return [node.id for node in result["route"]]


def test_algorithms_switch_route_after_incident():
    generator = ScenarioGraphGenerator()

    scenarios = {
        "NORMAL": generator.generate(),
        "INCIDENT": generator.upper_route_incident(),
        "BLOCKAGE": generator.upper_route_blockage(),
    }

    algorithms = {
        "Dijkstra": lambda graph: Dijkstra(graph),
        "PSO": lambda graph: PSO(
            graph,
            particles=20,
            iterations=50,
            seed=42,
            use_dijkstra_seed=True,
        ),
        "QPSO": lambda graph: QPSO(
            graph,
            particles=20,
            iterations=50,
            seed=42,
            use_dijkstra_seed=True,
        ),
    }

    for scenario_name, graph in scenarios.items():
        print(f"\n--- {scenario_name} ---")

        for algorithm_name, factory in algorithms.items():
            algorithm = factory(graph)

            result = algorithm.optimize("N0", "N5")

            route = route_ids(result)

            print(
                f"{algorithm_name}: "
                f"{' -> '.join(route)} | "
                f"fitness={result['fitness']:.4f} | "
                f"time={result['travel_time']:.4f}"
            )

            assert route[0] == "N0"
            assert route[-1] == "N5"

            if scenario_name in ("INCIDENT", "BLOCKAGE"):
                assert route == ["N0", "N3", "N4", "N5"]
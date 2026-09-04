from src.algorithms.warm_vrp_qpso import WarmVRPQPSO
from src.optimization.vrp_benchmark import VRPBenchmark


def create_problem():
    benchmark = VRPBenchmark(
        customer_sizes=(10,),
        particles=10,
        iterations=10,
        seed=42,
    )

    return benchmark._build_problem(10)


def test_warm_qpso_runs():

    problem = create_problem()

    optimizer = WarmVRPQPSO(
        problem=problem,
        particles=10,
        iterations=10,
        beta=0.7,
        seed=42,
        warm_iterations=5,
        warm_particles=5,
    )

    result = optimizer.optimize()

    assert result["valid"]
    assert result["solution"] is not None
    assert result["fitness"] > 0
    assert len(result["history"]) == 11


def test_warm_start_is_recorded():

    problem = create_problem()

    optimizer = WarmVRPQPSO(
        problem=problem,
        particles=10,
        iterations=5,
        beta=0.7,
        seed=42,
        warm_iterations=3,
        warm_particles=5,
    )

    result = optimizer.optimize()

    assert result["warm_start_fitness"] > 0
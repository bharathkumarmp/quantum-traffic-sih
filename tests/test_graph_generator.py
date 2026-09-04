from src.data.graph_generator import LargeGraphGenerator


def count_nodes(graph, expected_count):
    """
    Count generated nodes using the existing Graph API.

    Graph.get_node() raises KeyError when a node
    does not exist, so we check the known range directly.
    """

    count = 0

    for i in range(expected_count):
        try:
            graph.get_node(f"N{i}")
            count += 1
        except KeyError:
            pass

    return count


def test_small_graph_generation():

    generator = LargeGraphGenerator(
        num_nodes=10,
        edges_per_node=2,
        seed=42,
    )

    graph = generator.generate()

    assert count_nodes(graph, 10) == 10

    assert len(graph.get_edges()) > 0


def test_graph_is_reproducible():

    generator1 = LargeGraphGenerator(
        num_nodes=20,
        edges_per_node=2,
        seed=42,
    )

    generator2 = LargeGraphGenerator(
        num_nodes=20,
        edges_per_node=2,
        seed=42,
    )

    graph1 = generator1.generate()
    graph2 = generator2.generate()

    edges1 = graph1.get_edges()
    edges2 = graph2.get_edges()

    assert len(edges1) == len(edges2)

    for e1, e2 in zip(edges1, edges2):

        assert e1.source.id == e2.source.id
        assert e1.target.id == e2.target.id

        assert e1.distance == e2.distance
        assert e1.travel_time == e2.travel_time
        assert e1.congestion == e2.congestion


def test_large_graph_generation():

    generator = LargeGraphGenerator(
        num_nodes=1000,
        edges_per_node=3,
        seed=42,
    )

    graph = generator.generate()

    assert count_nodes(graph, 1000) == 1000

    assert len(graph.get_edges()) > 1000
class Graph:
    """
    Directed weighted graph representing a road network.
    """

    def __init__(
        self,
        nodes=None,
        edges=None,
    ):
        self.nodes = []
        self.edges = []

        self._node_map = {}

        if nodes:
            for node in nodes:
                self.add_node(node)

        if edges:
            for edge in edges:
                self.add_edge(edge)

    # =========================================================
    # ADD NODE
    # =========================================================

    def add_node(self, node):
        if node.id not in self._node_map:

            self.nodes.append(node)

            self._node_map[node.id] = node

    # =========================================================
    # ADD EDGE
    # =========================================================

    def add_edge(self, edge):

        if edge.source.id not in self._node_map:
            self.add_node(edge.source)

        if edge.target.id not in self._node_map:
            self.add_node(edge.target)

        self.edges.append(edge)

    # =========================================================
    # GET NODE
    # =========================================================

    def get_node(self, node_id):
        return self._node_map[str(node_id)]

    # =========================================================
    # GET EDGES
    # =========================================================

    def get_edges(self):
        return self.edges

    # =========================================================
    # GET NEIGHBORS
    # =========================================================

    def get_neighbors(self, node):
        return [
            edge
            for edge in self.edges
            if edge.source.id == node.id
        ]

    # =========================================================
    # NUMBER OF NODES
    # =========================================================

    def number_of_nodes(self):
        return len(self.nodes)

    # =========================================================
    # NUMBER OF EDGES
    # =========================================================

    def number_of_edges(self):
        return len(self.edges)

    def __repr__(self):
        return (
            f"Graph("
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}"
            f")"
        )
class Route:
    """
    Represents a candidate route through the transportation graph.
    """

    def __init__(self, nodes):
        self.nodes = list(nodes)

    @property
    def start(self):
        return self.nodes[0] if self.nodes else None

    @property
    def destination(self):
        return self.nodes[-1] if self.nodes else None

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        if not self.nodes:
            return "Route(empty)"

        return "Route(" + " -> ".join(
            node.id for node in self.nodes
        ) + ")"

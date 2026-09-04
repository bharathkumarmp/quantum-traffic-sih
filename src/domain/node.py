class Node:
    """
    Represents a node/intersection in the road network.
    """

    def __init__(self, node_id):
        self.id = str(node_id)

    def __repr__(self):
        return f"Node({self.id})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
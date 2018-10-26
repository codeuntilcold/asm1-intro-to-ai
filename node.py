'''
class Node:

    # Since our block can be in horizontal position, we need 2 X and 2 Y points
    # Initially, it does not have any successor
    # path_cost is the cost of reaching this node from initial node.
    def __init__(self, element, x, y, x2, y2, path_cost=0):
        self.element = element
        self.x = x
        self.x2 = x2
        self.y = y
        self.y2 = y2
        self.path_cost = path_cost
        self.successors = []

    def addSuccessor(self, node):
        self.successors.append(node)

    def getNodeInformation(self):
        return self.element, self.x, self.y, self.x2, self.y2
'''


# This code is taken from https://github.com/aimacode/aima-python/blob/master/search.py

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    # In this case, g --> path_cost  &   h --> distance_to_goal
    def __init__(self, x1, y1, x2, y2, parent=None, path_cost=0, distance_to_goal=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.parent = parent
        self.path_cost = path_cost
        self.distance_to_goal = distance_to_goal
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return self.x1 == other.x1 and self.x2 == other.x2 and self.y1 == other.y1 and self.y2 == other.y2

    def __lt__(self, other):
        return self.path_cost + self.distance_to_goal < other.path_cost + other.distance_to_goal

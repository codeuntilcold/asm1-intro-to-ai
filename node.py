# This code is taken from https://github.com/aimacode/aima-python/blob/master/search.py

class Node:
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
    
    def get_block_orientation(node):
        if node.x1 == node.x2 and node.y1 == node.y2:
            return "standing"
        elif abs(node.x1 - node.x2) == 1:
            return "vertical"
        else:
            return "horizontal"

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. 
    def __eq__(self, other):
        return self.x1 == other.x1 and self.x2 == other.x2 and self.y1 == other.y1 and self.y2 == other.y2

    def __lt__(self, other):
        return self.path_cost + self.distance_to_goal < other.path_cost + other.distance_to_goal

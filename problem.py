# This code is taken from https://github.com/aimacode/aima-python/blob/master/search.py
import math
from node import Node

# Get the coordinates of the Goal
def get_goal_points(board):
    x_goal, y_goal, index_board_row, index_board_col = -1, -1, -1, -1
    for rowBoard in board:
        index_board_row += 1
        for elementBoard in rowBoard:
            index_board_col += 1

            if elementBoard == "G":
                x_goal = index_board_row
                y_goal = index_board_col

        index_board_col = -1

    return x_goal, y_goal


# Get the coordinates of the Block
def get_block_points(board):
    x_start, y_start, x_end, y_end, index_board_row, index_board_col = -1, -1, -1, -1, -1, -1
    for rowBoard in board:
        index_board_row += 1
        for elementBoard in rowBoard:
            index_board_col += 1

            if elementBoard == "S":
                if x_start == -1:
                    x_start = index_board_row
                    y_start = index_board_col
                else:
                    x_end = index_board_row
                    y_end = index_board_col

        index_board_col = -1

    if x_end == -1:
        x_end = x_start
        y_end = y_start

    return x_start, y_start, x_end, y_end


class Problem(object):
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, board):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = []
        self.board = board
        self.x_goal, self.y_goal = get_goal_points(self.board)
        self.x_block_1, self.y_block_1, self.x_block_2, self.y_block_2 = get_block_points(self.board)
        self.number_of_rows = len(board)
        self.number_of_columns = len(board[0])
        self.walker = [ self.check_left, self.check_down, self.check_up, self.check_right ]

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, node):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if node.x1 == node.x2 and node.x2 == self.x_goal and node.y1 == node.y2 and node.y2 == self.y_goal:
            return True

        return False

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

    def get_block_points(self):
        return self.x_block_1, self.y_block_1, self.x_block_2, self.y_block_2

    def find_distance_to_goal(self, node):
        x_avg = (node.x1 + node.x2) / 2
        y_avg = (node.y1 + node.y2) / 2

        hypotenuse = math.sqrt(math.pow((self.x_goal - x_avg), 2) + math.pow((self.y_goal - y_avg), 2))

        return hypotenuse
    
    # Check if we can move right
    def check_right(self, node, orientation):
        current_path_cost = node.path_cost
        distance_to_new_node = 0

        if orientation == "horizontal":
            y_new_pos1 = node.y1 + 2
            y_new_pos2 = node.y2 + 1
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 3

            if y_new_pos1 < self.number_of_columns:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

            return None

        elif orientation == "vertical":
            y_new_pos1 = node.y1 + 1
            y_new_pos2 = node.y2 + 1
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 2

            if y_new_pos1 < self.number_of_columns:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

        else:
            y_new_pos1 = node.y1 + 1
            y_new_pos2 = node.y2 + 2
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 3

            if y_new_pos2 < self.number_of_columns:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)


    # Check if we can move left
    def check_left(self, node, orientation):
        current_path_cost = node.path_cost
        distance_to_new_node = 0

        if orientation == "horizontal":
            y_new_pos1 = node.y1 - 1
            y_new_pos2 = node.y2 - 2
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 3

            if y_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

            return None

        elif orientation == "vertical":
            y_new_pos1 = node.y1 - 1
            y_new_pos2 = node.y2 - 1
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 2

            if y_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

        else:
            y_new_pos1 = node.y1 - 2
            y_new_pos2 = node.y2 - 1
            x_new_pos1 = node.x1
            x_new_pos2 = node.x2

            distance_to_new_node = 3

            if y_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)


    # Check if we can move down
    def check_down(self, node, orientation):
        current_path_cost = node.path_cost
        distance_to_new_node = 0

        if orientation == "horizontal":
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 + 1
            x_new_pos2 = node.x2 + 1

            distance_to_new_node = 2

            if x_new_pos1 < self.number_of_rows:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

            return None

        elif orientation == "vertical":
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 + 2
            x_new_pos2 = node.x2 + 1

            distance_to_new_node = 3

            if x_new_pos1 < self.number_of_rows:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

        else:
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 + 1
            x_new_pos2 = node.x2 + 2

            distance_to_new_node = 3

            if x_new_pos2 < self.number_of_rows:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)


    # Check if we can move up
    def check_up(self, node, orientation):
        current_path_cost = node.path_cost
        distance_to_new_node = 0

        if orientation == "horizontal":
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 - 1
            x_new_pos2 = node.x2 - 1

            distance_to_new_node = 2

            if x_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

            return None

        elif orientation == "vertical":
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 - 1
            x_new_pos2 = node.x2 - 2

            distance_to_new_node = 3

            if x_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

        else:
            y_new_pos1 = node.y1
            y_new_pos2 = node.y2
            x_new_pos1 = node.x1 - 2
            x_new_pos2 = node.x2 - 1

            distance_to_new_node = 3

            if x_new_pos1 >= 0:
                if self.board[x_new_pos1][y_new_pos1] != "X" and self.board[x_new_pos2][y_new_pos2] != "X":
                    return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                                path_cost=current_path_cost + distance_to_new_node)

    def walk_along(self, step, node):
        return self.walker[step](node, node.get_block_orientation())

from problem import Problem
from node import Node
from min_heap import PriorityQueue
from utils import memoize
import time
import resource


# Ege Alpay 19551


# Get coordinates of the Goal
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


# Check coordinates of the block
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


# Check block orientation
def get_block_orientation(node):
    if node.x1 == node.x2 and node.y1 == node.y2:
        return "vertical"
    elif node.x1 + 1 == node.x2 or node.x2 + 1 == node.x1:
        return "horizontal_north_south"
    else:
        return "horizontal_east_west"


# Check successor on right
def check_right(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal_east_west":
        y_new_pos1 = node.y1 + 2
        y_new_pos2 = node.y2 + 1
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 3

        if y_new_pos1 < number_of_columns:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

        return None

    elif orientation == "horizontal_north_south":
        y_new_pos1 = node.y1 + 1
        y_new_pos2 = node.y2 + 1
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 2

        if y_new_pos1 < number_of_columns:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

    else:
        y_new_pos1 = node.y1 + 1
        y_new_pos2 = node.y2 + 2
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 3

        if y_new_pos2 < number_of_columns:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)


# Check successor on left
def check_left(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal_east_west":
        y_new_pos1 = node.y1 - 1
        y_new_pos2 = node.y2 - 2
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 3

        if y_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

        return None

    elif orientation == "horizontal_north_south":
        y_new_pos1 = node.y1 - 1
        y_new_pos2 = node.y2 - 1
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 2

        if y_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

    else:
        y_new_pos1 = node.y1 - 2
        y_new_pos2 = node.y2 - 1
        x_new_pos1 = node.x1
        x_new_pos2 = node.x2

        distance_to_new_node = 3

        if y_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)


# Check successor below
def check_down(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal_east_west":
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 + 1
        x_new_pos2 = node.x2 + 1

        distance_to_new_node = 2

        if x_new_pos1 < number_of_rows:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

        return None

    elif orientation == "horizontal_north_south":
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 + 2
        x_new_pos2 = node.x2 + 1

        distance_to_new_node = 3

        if x_new_pos1 < number_of_rows:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

    else:
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 + 1
        x_new_pos2 = node.x2 + 2

        distance_to_new_node = 3

        if x_new_pos2 < number_of_rows:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)


# Check successor on top
def check_up(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal_east_west":
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 - 1
        x_new_pos2 = node.x2 - 1

        distance_to_new_node = 2

        if x_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

        return None

    elif orientation == "horizontal_north_south":
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 - 1
        x_new_pos2 = node.x2 - 2

        distance_to_new_node = 3

        if x_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)

    else:
        y_new_pos1 = node.y1
        y_new_pos2 = node.y2
        x_new_pos1 = node.x1 - 2
        x_new_pos2 = node.x2 - 1

        distance_to_new_node = 3

        if x_new_pos1 >= 0:
            if sample_matrix[x_new_pos1][y_new_pos1] != "X" and sample_matrix[x_new_pos2][y_new_pos2] != "X":
                return Node(x_new_pos1, y_new_pos1, x_new_pos2, y_new_pos2, parent=node,
                            path_cost=current_path_cost + distance_to_new_node)


# Finding Successors
def find_successors(node):
    orientation = get_block_orientation(node)
    successor_nodes = []

    # For each orientation, there are 4 different actions: Up, Down, Left, Right
    up_successor = check_up(node, orientation)
    down_successor = check_down(node, orientation)
    left_successor = check_left(node, orientation)
    right_successor = check_right(node, orientation)

    if up_successor is not None:
        successor_nodes.append(up_successor)
    if down_successor is not None:
        successor_nodes.append(down_successor)
    if left_successor is not None:
        successor_nodes.append(left_successor)
    if right_successor is not None:
        successor_nodes.append(right_successor)

    return successor_nodes


def best_first_graph_search(problem, f, x_block_1, y_block_1, x_block_2, y_block_2):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    # Our initial node / state
    initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2)
    # Priority queue will give our the node which has the lowest f value
    frontier = PriorityQueue('min', f)
    frontier.append(initial_node)
    explored = set()
    # While there is a node in frontier:
    while frontier:
        # Take node from frontier
        node = frontier.pop()
        # Check if this node is the goal node
        if problem.goal_test(node):
            # return the goal node and memory consumption
            return node, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Add it to explored set
        explored.add((node.x1, node.y1, node.x2, node.y2))
        # Find the successors of the current node
        successors = find_successors(node)
        # For each successor of the current node
        for successor in successors:
            # If the successor has not visited and is not in frontier
            if (successor.x1, successor.y1, successor.x2, successor.y2) not in explored and successor not in frontier:
                # Add it to frontier
                frontier.append(successor)
            # If it is in frontier
            elif successor in frontier:
                # Compare f values and update if needed
                incumbent = frontier[successor]
                if f(successor) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(successor)
    # Return None if we can't reach to goal node
    return None, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def uniform_cost_search(problem, x_block_1, y_block_1, x_block_2, y_block_2):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, x_block_1, y_block_1, x_block_2, y_block_2)


'''
sample_matrix = [
    ['O', 'O', 'O', 'X', 'O', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'G', 'X'],
    ['X', 'X', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['S', 'S', 'O', 'X', 'X', 'X', 'O', 'O']
]
'''
'''
sample_matrix = [
    ["O", "O", "O", "X", "X", "X", "X", "X", "X", "X"],
    ["O", "O", "O", "O", "O", "O", "X", "X", "X", "X"],
    ["O", "O", "O", "S", "O", "O", "O", "O", "O", "X"],
    ["X", "O", "O", "S", "O", "O", "O", "O", "O", "O"],
    ["X", "X", "X", "X", "X", "O", "O", "G", "O", "O"],
    ["X", "X", "X", "X", "X", "X", "O", "O", "O", "X"],
]
'''
'''
sample_matrix = [
    ["S", "O", "O"],
    ["O", "O", "O"],
    ["O", "O", "G"],
    ["X", "O", "O"],
    ["X", "X", "X"],
]
'''
'''
sample_matrix = [
    ['O', 'O', 'O', 'X', 'O', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'G', 'X'],
    ['X', 'X', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['S', 'S', 'O', 'X', 'X', 'X', 'O', 'O']
]
'''
sample_matrix = [
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['S', 'S', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X']
]

# Number of rows and columns of the matrix
number_of_rows = len(sample_matrix)
number_of_columns = len(sample_matrix[0])

# Coordinates of the goal
x_goal, y_goal = get_goal_points(sample_matrix)

# Coordinates of the block
x_block_1, y_block_1, x_block_2, y_block_2 = get_block_points(sample_matrix)

# Initialized our problem with the coordinates of the goal
problem = Problem(x_goal, y_goal)

# Measure time
start_time = time.time()
result_node, memory_info = uniform_cost_search(problem, x_block_1, y_block_1, x_block_2, y_block_2)
end_time = time.time()

print("\n--- %s seconds ---" % (end_time - start_time))
print(f"\n{memory_info} bytes\n")

# If no solution exists
if result_node is None:
    print("Solution does not exists")
else:
    # If solution exists
    # It is -1 since "count_of_moves = number_of_nodes - 1"
    count_of_moves = -1

    print("\nStarting from coordinates of the goal to coordinates of the initial block: ")
    while result_node is not None:
        print(f"(({result_node.x1}, {result_node.y1}), ({result_node.x2}, {result_node.y2}))")
        print("\n")
        result_node = result_node.parent
        count_of_moves += 1

    print(f"Total moves needed: {count_of_moves}")

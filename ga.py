import random

random.seed(32)

from random import randint, random
from problem import Problem
from node import Node
import math
import time
import resource
import numpy as np


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


# To understand the orientation of the node
def get_block_orientation(node):
    if node.x1 == node.x2 and node.y1 == node.y2:
        return "standing"
    elif abs(node.x1 - node.x2) == 1:
        return "vertical"
    else:
        return "horizontal"


# Check if we can move right
def check_right(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal":
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

    elif orientation == "vertical":
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


# Check if we can move left
def check_left(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal":
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

    elif orientation == "vertical":
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


# Check if we can move down
def check_down(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal":
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

    elif orientation == "vertical":
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


# Check if we can move up
def check_up(node, orientation):
    current_path_cost = node.path_cost
    distance_to_new_node = 0

    if orientation == "horizontal":
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

    elif orientation == "vertical":
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


# This method represents h(n). Calculates the distance from current position to goal
def find_distance_to_goal(node):
    x_avg = (node.x1 + node.x2) / 2
    y_avg = (node.y1 + node.y2) / 2

    hypotenuse = math.sqrt(math.pow((x_goal - x_avg), 2) + math.pow((y_goal - y_avg), 2))

    return hypotenuse


### GENETIC ALGORITHM ###

def crossover(p1, p2, r_cross):
    if random() < r_cross:
        parent1, parent2 = p1.copy(), p2.copy()
        crossover_point = randint(1, len(p1)-2)
        c1 = parent1[:crossover_point] + parent2[crossover_point:]
        c2 = parent2[:crossover_point] + parent1[crossover_point:]
        return [c1, c2]

def mutation(path, r_mut):
    if random() < r_mut:
        new_path = path.copy()
        new_path[randint(0, len(path) - 1)] = randint(0, 3) 
        return new_path

walker = [ check_left, check_down, check_up, check_right ]
def walk(problem, node, path):
    if problem.goal_test(node):
        return True, 0
    if len(path) == 0:
        return False, node.distance_to_goal

    node.distance_to_goal = find_distance_to_goal(node) + (1 - int(problem.goal_test(node)))
    step = path[0]
    orietation = get_block_orientation(node)
    successor = walker[step](node, orietation)
    if successor is None:
        return False, node.distance_to_goal
    return walk(problem, successor, path[1:])

def genetic_algorithm_search(problem, x_block_1, y_block_1, x_block_2, y_block_2):
    MAX_PATH = 2 * (number_of_rows + number_of_columns)
    POPULATION_SIZE = 10
    MAX_GENERATIONS = 10
    P_CROSSOVER = 0.75
    
    paths = [[randint(0, 3) for _ in range(MAX_PATH)] for _ in range(POPULATION_SIZE)]

    for i in range(MAX_GENERATIONS):
        print(f"GENERATION {i+1}===")
        # Calculate value of each node after stepping
        results = []
        for path in paths:
            initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2)
            success, distance = walk(problem, initial_node, path)
            if success:
                return path, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            else:
                results.append(distance)

        # Reproduce
        for i in range(0, POPULATION_SIZE, 2):
            p1, p2 = paths[i], paths[i+1]
            childs = crossover(p1, p2, P_CROSSOVER)
            if childs:
                for path in childs:
                    initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2)
                    success, distance = walk(problem, initial_node, path)
                    if success:
                        return path, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    else:
                        paths.append(path)
                        results.append(distance)

        # Mutation
        for path in paths:
            child = mutation(path, i / float(MAX_GENERATIONS))
            if child:
                initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2)
                success, distance = walk(problem, initial_node, child)
                if success:
                    return child, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                else:
                    paths.append(child)
                    results.append(distance)

        best_idx = np.argpartition(np.array(results), POPULATION_SIZE)[:POPULATION_SIZE]
        results = [results[i] for i in best_idx]
        paths = [paths[i] for i in best_idx]
        print(f"Best result is {min(results)}")

        # print(f"Best results until now: {min([results[i] for i in best_idx])}")
    
    # Cannot find path
    return [], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


        
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


number_of_rows = len(sample_matrix)
number_of_columns = len(sample_matrix[0])

# Coordinates of the goal
x_goal, y_goal = get_goal_points(sample_matrix)

# Coordinates of the block
x_block_1, y_block_1, x_block_2, y_block_2 = get_block_points(sample_matrix)

problem = Problem(x_goal, y_goal)

start_time = time.time()
result_node, memory_info = genetic_algorithm_search(problem, x_block_1, y_block_1, x_block_2, y_block_2)
end_time = time.time()

print("\n--- %s seconds ---" % (end_time - start_time))
print(f"\n{memory_info} bytes\n")

if result_node is None:
    print("Solution does not exists")
else:
    mapping = [ "LEFT", "DOWN", "UP", "RIGHT" ]
    print(list(map(lambda step: mapping[step], result_node)))

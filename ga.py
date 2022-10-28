import random

random.seed(32)

from random import randint, random
from problem import Problem
from node import Node
import time
import resource
import numpy as np
import json

class GeneticAlgorithm:
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

    def walk(problem, node, path):
        if problem.goal_test(node):
            return True, node
        if len(path) == 0:
            return False, node

        node.distance_to_goal = problem.find_distance_to_goal(node) + (1 - int(problem.goal_test(node)))
        successor = problem.walk_along(path[0], node)
        if successor is None:
            return False, node
        return GeneticAlgorithm.walk(problem, successor, path[1:])

    def run(problem: Problem):
        x_block_1, y_block_1, x_block_2, y_block_2 = problem.get_block_points()
        MAX_PATH = 2 * (problem.number_of_rows + problem.number_of_columns)
        POPULATION_SIZE = 10
        MAX_GENERATIONS = 10
        P_CROSSOVER = 0.75
        
        paths = [[randint(0, 3) for _ in range(MAX_PATH)] for _ in range(POPULATION_SIZE)]

        for i in range(MAX_GENERATIONS):
            print(f"GENERATION {i+1}===")
            # Calculate value of each node after stepping
            results = []
            for path in paths:
                initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2, -1)
                success, node = GeneticAlgorithm.walk(problem, initial_node, path)
                if success:
                    return node.solution(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                else:
                    results.append(node.distance_to_goal)

            # Reproduce
            for i in range(0, POPULATION_SIZE, 2):
                p1, p2 = paths[i], paths[i+1]
                childs = GeneticAlgorithm.crossover(p1, p2, P_CROSSOVER)
                if childs:
                    for path in childs:
                        initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2, -1)
                        success, node = GeneticAlgorithm.walk(problem, initial_node, path)
                        if success:
                            return node.solution(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        else:
                            paths.append(path)
                            results.append(node.distance_to_goal)

            # Mutation
            for path in paths:
                child = GeneticAlgorithm.mutation(path, i / float(MAX_GENERATIONS))
                if child:
                    initial_node = Node(x_block_1, y_block_1, x_block_2, y_block_2, -1)
                    success, node = GeneticAlgorithm.walk(problem, initial_node, child)
                    if success:
                        return node.solution(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    else:
                        paths.append(child)
                        results.append(node.distance_to_goal)

            best_idx = np.argpartition(np.array(results), POPULATION_SIZE)[:POPULATION_SIZE]
            results = [results[i] for i in best_idx]
            paths = [paths[i] for i in best_idx]
            print(f"Best result is {min(results)}")

            # print(f"Best results until now: {min([results[i] for i in best_idx])}")
        
        # Cannot find path
        return [], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


if __name__ == '__main__':
    with open("maps.json", "r") as f:
        sample_matrix = json.load(f)

    # Note: Not all maps are solvable
    problem = Problem(sample_matrix[0])
    start_time = time.time()
    solution, memory_info = GeneticAlgorithm.run(problem)
    end_time = time.time()

    print("\n--- %s seconds ---" % (end_time - start_time))
    print(f"\n{memory_info} bytes\n")

    if solution is None:
        print("Cannot find solution.")
    else:
        step_map = [ "LEFT", "DOWN", "UP", "RIGHT" ]
        print(f"Original solution:\t{list(map(lambda x: step_map[x], solution))}")
        
        short_solution = solution.copy()
        i = 0
        while i < len(short_solution[:-1]):
            if short_solution[i] + short_solution[i + 1] == 3:
                del short_solution[i:i+2]
            i += 1

        print(f"Redundant-free:\t\t{list(map(lambda x: step_map[x], short_solution))}")

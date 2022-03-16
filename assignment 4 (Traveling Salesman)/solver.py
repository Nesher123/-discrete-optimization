#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from collections import namedtuple
import sys
import logging
import numpy as np
import json5 as json
import random
import copy
from itertools import combinations

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
Point = namedtuple('Point', ['x', 'y', 'i'])

"""
Helper functions
"""


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_length_of_tour(points: list[Point], solution: list[int]) -> float:
    """calculate the length of the tour"""
    objective = length(points[solution[-1]], points[solution[0]])

    for index in range(0, len(solution) - 1):
        objective += length(points[solution[index]], points[solution[index + 1]])

    return objective


def get_random_indices(my_list: list[int]) -> [int, int]:
    start_index, end_index = -1, -1

    while start_index == end_index:
        start_index, end_index = sorted(
            [random.randint(0, len(my_list) - 1), random.randint(0, len(my_list) - 1)])

    return start_index, end_index


"""
Different solutions functions
"""


def trivial(node_count: int) -> list[int]:
    """
    build a trivial solution
    visit the nodes in the order they appear in the file
    """
    return list(range(0, node_count))


def greedy(points: list[Point]) -> list[int]:
    """
    greedy approach:
    at every step, go to the nearest point.
    """
    solution = [0]
    original_length = len(points)
    j = 0
    next_vertex = points[0]
    points.pop(0)

    while j < original_length - 1:
        x, y, i = next_vertex
        closest = np.argmin([(p.x - x) ** 2 + (p.y - y) ** 2 for p in points if p.i != i])
        next_vertex = points[closest]
        solution.append(next_vertex.i)
        points.pop(closest)
        j = j + 1

    return solution


def simulated_annealing(points: list[Point], solution: list[int], config_data: dict, approach: str,
                        initial_temperature: int, number_of_iterations: int) -> [
    list[int], float]:
    """
    Get a solution (can be random or the results of the greedy algorithm, as a better baseline).
    Then, apply simulated annealing algorithm on the given solution to improve the total cost (length of tour)

    We can apply either a reverse/transport/swap approach:
        - reverse the tour between 2 indices
        - move a part between 2 indices to the end of the solution and change the original solution anyway
        - swap values of 2 indices such that the tour is changed

    :param points:
    :param solution: current working solution
    :param config_data:
    :param approach:
    :param initial_temperature:
    :param number_of_iterations:

    :return: best working solution
    """
    best_objective = calculate_length_of_tour(points, solution)  # evaluate the initial solution

    for i in range(number_of_iterations):
        temperature = initial_temperature / float(i + 1)  # calculate temperature for current epoch

        start_index, end_index = get_random_indices(solution)

        candidate_solution = copy.deepcopy(solution)

        if approach == config_data['reverse']:
            candidate_solution[start_index:end_index + 1] = list(
                reversed(candidate_solution[start_index:end_index + 1]))
        elif approach == config_data['transport']:
            candidate_solution = solution[0:start_index] + solution[end_index:] + list(solution[start_index:end_index])
        elif approach == config_data['swap']:  # randomly swap 2 vertices
            temp = candidate_solution[start_index]
            candidate_solution[start_index] = candidate_solution[end_index]
            candidate_solution[end_index] = temp
        else:
            raise ValueError('Provide a valid approach name in the configuration file, under the "approach" attribute.')

        candidate_objective = calculate_length_of_tour(points, candidate_solution)
        diff = candidate_objective - best_objective  # difference between candidate and current evaluations

        x = min(-diff / temperature, 10)
        metropolis = np.exp(x)  # calculate metropolis acceptance criterion

        # check for new best solution or metropolis acceptance criterion satisfied
        if (diff < 0) | (np.random.rand() < metropolis):
            best_objective = candidate_objective
            solution = candidate_solution  # store it

    return solution, calculate_length_of_tour(points, solution)


def two_opt_swap(points: list[Point], solution: list[int], start_index: int, end_index: int,
                 improvement_threshold: float = 10 ** -6) -> bool:
    """
    Check if swapping 2 indices (given as start_index and end_index) reduces the tour's length

    :param points:
    :param solution:
    :param start_index:
    :param end_index:
    :param improvement_threshold:

    :return: bool
        True if swapping improves
        False otherwise
    """
    improved = False
    partial_objective = calculate_length_of_tour(points, solution[start_index:end_index])
    candidate_solution = solution[:start_index] + solution[start_index:end_index + 1][::-1] + solution[end_index + 1:]
    new_partial_objective = partial_objective - \
                            length(points[solution[start_index - 1]], points[solution[start_index]]) - \
                            length(points[solution[end_index]], points[solution[end_index + 1]]) + \
                            length(points[candidate_solution[start_index - 1]],
                                   points[candidate_solution[start_index]]) + \
                            length(points[candidate_solution[end_index]], points[candidate_solution[(end_index + 1)]])

    if new_partial_objective < partial_objective - improvement_threshold:
        improved = True

    return improved


def two_opt(points: list[Point], solution: list[int]) -> list[int]:
    """
    Check EVERY pair of indices and override current solution if swapping returns True.

    :param points:
    :param solution:
    :return: the best solution after all swaps occurred (if any)
    """
    improved = True

    while improved:
        improved = False

        for start_index, end_index in combinations(range(1, len(solution) - 1), 2):
            if two_opt_swap(points, solution, start_index, end_index):
                solution = solution[:start_index] + solution[start_index:end_index + 1][::-1] + solution[end_index + 1:]
                improved = True
                break

    return solution


def shuffle(input_list, n=2):
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(input_list)), k=n)
    to_shuffle = [input_list[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)

    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        input_list[old_index] = value

    return input_list


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    node_count = int(lines[0])

    points = []

    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i - 1))

    config_data = json.load(open('config.json5', 'r'))
    is_optimal = False

    try:
        with open(f'recent_solutions/solution_{node_count}.txt', 'r') as _:
            solution_lines = _.read().split('\n')
            solution = list(map(int, solution_lines[1].split(' ')))
            objective = calculate_length_of_tour(points, solution)
    except FileNotFoundError:
        # choose your desired algorithm to run:
        # solution = trivial(node_count)
        solution = greedy(points.copy())

    solution = trivial(node_count)
    new_objective = calculate_length_of_tour(points, solution)
    BEST = config_data[f'solution_{node_count}']

    while BEST < new_objective:
        # approach = random.choice([config_data['reverse'], config_data['transport']])
        # solution, new_objective = simulated_annealing(
        #     points, solution, config_data, approach, config_data['initial_temperature'],
        #     config_data['number_of_iterations'])
        #
        # # we can perform ANOTHER round of simulated_annealing with different/same approach to help improve the result
        #
        # logging.info(f'1: {new_objective}')
        #
        # if BEST >= new_objective:
        #     break

        # apply 2-OPT swap algorithm to improve results even more
        solution = two_opt(points, solution)
        new_objective = calculate_length_of_tour(points, solution)
        logging.info(f'2: {new_objective}')

        if BEST < new_objective:
            # local minima encountered - shuffle parts of the solution in order to escape it.
            solution = shuffle(solution, n=config_data['shuffle_amount'])

    # new record for this problem set!
    objective = new_objective

    # prepare the solution in the specified output format
    output_data = '%.2f' % objective + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

    with open(f'solution_{node_count}.txt', 'w') as f:
        f.write(output_data)

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()

        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        print(solve_it(input_data))
    else:
        print('This test requires an input file. '
              'Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

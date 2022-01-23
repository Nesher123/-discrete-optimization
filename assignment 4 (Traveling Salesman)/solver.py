#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import sys
import logging
import numpy as np
import json5 as json
import random

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


def simulated_annealing(points: list[Point], solution: list[int]) -> [list[int], float]:
    """
    get a solution (can be random or the results of the greedy algorithm, for a better baseline)
    Then, apply simulated annealing algorithm on the given solution to improve the total cost (length of the road)

    :param points:
    :param solution: current working solution
    :return: best working solution
    """
    number_of_iterations = 1000
    initial_temperature = 100  # initial temperature
    best_objective = calculate_length_of_tour(points, solution)  # evaluate the initial solution

    for i in range(number_of_iterations):
        temperature = initial_temperature / float(i + 1)  # calculate temperature for current epoch
        start_index, end_index = sorted(random.sample(range(0, len(solution)), 2))
        candidate_solution = \
            solution[0:start_index] + list(reversed(solution[start_index:end_index])) + solution[end_index:]
        # candidate_solution = solution[0:start_index] + solution[end_index:] + list(solution[start_index:end_index])
        candidate_objective = calculate_length_of_tour(points, candidate_solution)

        diff = candidate_objective - best_objective  # difference between candidate and current evaluations
        metropolis = np.exp(-diff / temperature)  # calculate metropolis acceptance criterion

        # check for new best solution or metropolis acceptance criterion satisfied
        if (diff < 0) | (np.random.rand() < metropolis):
            best_objective = candidate_objective
            solution = candidate_solution  # store it

    return solution, best_objective


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

    # choose your desired algorithm to run:
    # solution = trivial(node_count)
    solution = greedy(points.copy())
    solution, objective = simulated_annealing(points, solution)
    is_optimal = False

    # prepare the solution in the specified output format
    output_data = '%.2f' % objective + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

    # with open('solution.txt', 'w') as f:
    #     f.write(output_data)

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

#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import sys
import matplotlib.pyplot as plt
import logging
import json5 as json
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status, MiniZincError
import numpy as np

logging.basicConfig(level=logging.INFO)
Point = namedtuple('Point', ['x', 'y', 'i'])

"""
Helper functions
"""


def plot_points(points: list[Point]) -> None:
    x = [p[0] for p in points]  # x-axis values
    y = [p[1] for p in points]  # y-axis values

    plt.figure()

    for p in points:
        plt.annotate(p.i, (p.x, p.y))

    plt.scatter(x, y)  # plotting points as a scatter plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot')
    plt.show()


def get_ordered_list(points: list[Point], x, y) -> list[Point]:
    # points = [Point(x=0.0, y=0.0, i=0), Point(x=1.0, y=1.0, i=1), Point(x=0.0, y=0.5, i=2)]
    points.sort(key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)

    return points


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


"""
Different solutions functions
"""


def trivial(node_count: int) -> range:
    """
    build a trivial solution
    visit the nodes in the order they appear in the file
    """
    return range(0, node_count)


def greedy(points: list[Point]) -> list[int]:
    """
    greedy approach:
    at every step, go to the nearest point.
    """
    solution = [0]
    original_length = len(points)
    j = 0
    next = points[0]
    points.pop(0)

    while j < original_length - 1:
        x, y, i = next
        closest = np.argmin([(p.x - x) ** 2 + (p.y - y) ** 2 for p in points if p.i != i])
        next = points[closest]
        solution.append(next.i)
        points.pop(closest)
        j = j + 1

    return solution


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

    # plot_points(points)
    # solution, is_optimal = trivial(node_count), False
    solution, is_optimal = greedy(points.copy()), False

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])

    for index in range(0, node_count - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

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

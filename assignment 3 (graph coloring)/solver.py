#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Helper functions"""
import logging


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


"""Different solutions functions"""


def trivial(node_count: int) -> range:
    """
    build a trivial solution
    every node has its own color
    """
    return range(0, node_count)


def greedy(node_count: int, edges: list[tuple[int, int]]) -> list[int]:
    """
    greedy approach:
    all vertices are given color i.
    then, loop on the vertices, and assign color i+1 to every adjacent node
    """
    solution = [0] * node_count

    for i, j in edges:
        if solution[j] == solution[i]:
            solution[j] = solution[i] + 1

    for i in range(0, node_count):
        adjacent_vertices = find_indices(solution, lambda e: e == i)
        import itertools

        a = [adjacent_vertices] * 2
        edges_inner = tuple(itertools.product(*a))

        for e in edges_inner:
            if e in edges:
                solution[e[1]] = solution[e[1]] + 1

    return solution


def CP_solver(num_vals, edges) -> [list[int], bool]:
    """calling the solver"""
    from ortools.sat.python import cp_model

    # Create the model
    model = cp_model.CpModel()

    # Create the variables:
    variables = [
        model.NewIntVar(0, num_vals - 1, f'v_{i}') for i in range(num_vals)
    ]

    # Create the constraints:
    [model.Add(variables[v_1] != variables[v_2]) for v_1, v_2 in edges]

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # print_solution
    if (status == cp_model.OPTIMAL) | (status == cp_model.FEASIBLE):
        solution = [solver.Value(i) for i in variables]
    else:
        raise ValueError('No solution found.')

    return solution, (status == cp_model.OPTIMAL)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    is_optimal = False

    edges = []

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # solution = trivial(node_count)
    # solution = greedy(node_count, edges)
    solution, is_optimal = CP_solver(node_count, edges)

    objective_value = len(set(solution))  # number of colors

    # prepare the solution in the specified output format
    output_data = str(objective_value) + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()

        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. '
              '(i.e. python solver.py ./data/gc_4_1)')

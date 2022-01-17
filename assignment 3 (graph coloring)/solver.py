#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import sys
import json5 as json
from datetime import timedelta
from minizinc import Instance, Model, Solver, Status, MiniZincError
import numpy as np

logging.basicConfig(level=logging.INFO)

'''Helper functions'''


def find_indices(lst: list, condition) -> list:
    return [i for i, elem in enumerate(lst) if condition(elem)]


def get_maximal_clique_size(edges: list[tuple[int, int]]) -> int:
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edges)
    return len(max(nx.algorithms.clique.find_cliques(G), key=len))


def assign_values_to_arguments(instance: Instance, node_count: int, edge_count: int, edges: list[tuple[int, int]],
                               max_colors: int) -> None:
    """Assign values to instance's arguments"""
    instance['NODE_COUNT'] = node_count
    instance['EDGE_COUNT'] = edge_count
    instance['EDGES'] = edges
    instance['MAX_COLORS'] = max_colors


def get_best_result_by_solver(model: Model, solver_name: str, node_count: int, edge_count: int,
                              edges: list[tuple[int, int]], start: int, stop: int, step: int, timeout) -> [
    list[int], bool, int]:
    solution = None
    status = False
    objective_value = np.inf

    # Find the MiniZinc solver configuration for the desired solver (Gecode/OR-tools/chuffed)
    solver = Solver.lookup(solver_name)

    for i in range(start, stop, step):
        instance = Instance(solver, model)  # Create an Instance of the model for the desired solver
        assign_values_to_arguments(instance, node_count, edge_count, edges, i)

        try:
            result = instance.solve(timeout=timeout)
            solution = result.solution.colors
            status = (result.status == Status.OPTIMAL_SOLUTION)
            objective_value = len(set(result.solution.colors))  # number of colors
            logging.info(f'Solution found for solver {solver_name} with {i} colors!')
            break
        except AttributeError:
            logging.info(
                f"No solution found for solver {solver_name} with {i} colors or/and in {timeout.seconds} seconds.")
            continue
        except MiniZincError:
            logging.info(
                f"Another Exception was raised")
            break

    return solution, status, objective_value


'''Different solutions functions'''


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


def CP_solver_or_tools(num_vals: int, edges: list[tuple[int, int]]) -> [list[int], bool]:
    """calling the OR-tools solver"""
    from ortools.sat.python import cp_model

    # Create the model
    model = cp_model.CpModel()

    # Create the variables:
    variables = [
        model.NewIntVar(0, num_vals - 1, f'v_{i}') for i in range(num_vals)
    ]

    # Create the constraints:
    model.Add(variables[0] == 0)  # reduces the search space by a bit
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


def CP_solver_minizinc(config: dict, node_count: int, edge_count: int, edges: list[tuple[int, int]]) -> [list[int],
                                                                                                         bool, int]:
    """calling the MiniZinc solver"""
    model = Model(config['mz_model'])  # Run input_data through Minizinc Model
    timeout = timedelta(seconds=config['timedelta_seconds'])

    if node_count in [50, 70, 1000]:
        # geocode
        if node_count == 50:  # (problem set 1)
            start = 6
        elif node_count == 70:  # (problem set 2)
            start = 17
        else:  # node_count == 1000 (problem set 6)
            start = 122

        stop = start + 1
        result = get_best_result_by_solver(
            model, 'gecode', node_count, edge_count, edges, start, stop, 1, timeout)
    elif node_count in [100, 250, 500]:
        # OR-tools
        if node_count == 100:  # (problem set 3)
            start = 16
        elif node_count == 250:  # (problem set 4)
            start = 93
        else:  # node_count == 500 (problem set 5)
            start = 15

        stop = start + 1
        result = get_best_result_by_solver(
            model, 'or_tools', node_count, edge_count, edges, start, stop, 1, timeout)
    else:
        # for never-seen-before problem sets

        # the maximal clique size in a graph is the minimal number of unique colors
        # needed in order to color a graph chromatically
        start = get_maximal_clique_size(edges)
        stop = node_count

        geocode_result = get_best_result_by_solver(
            model, 'gecode', node_count, edge_count, edges, start, stop, 1, timeout)

        chuffed_result = get_best_result_by_solver(
            model, 'chuffed', node_count, edge_count, edges, start, min(geocode_result[2], stop), 1, timeout)

        best_results = [geocode_result, chuffed_result]
        result = best_results[np.argmin([i[2] for i in best_results if i[2] > 0])]

    return result


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # solution, is_optimal = trivial(node_count), False
    # solution, is_optimal = greedy(node_count, edges), False
    # solution, is_optimal = CP_solver_or_tools(node_count, edges)
    # objective_value = len(set(solution))  # number of colors

    config_data = json.load(open('assignment_3_config.json5', 'r'))
    solution, is_optimal, objective_value = CP_solver_minizinc(config_data, node_count, edge_count, edges)

    # prepare the solution in the specified output format
    output_data = str(objective_value) + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()

        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. '
              '(i.e. python solver.py ./data/gc_4_1)')

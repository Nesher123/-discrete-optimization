#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys

Item = namedtuple("Item", ['index', 'value', 'weight'])


def add_items_from_list_by_order_until_full(input_list: list, capacity: int) -> [float, float]:
    """given a list of items, add them to the knapsack by their order until capacity is reached"""
    value = 0
    weight = 0
    taken = [0] * len(input_list)

    for item in input_list:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


def naive(items: list, capacity: int) -> [float, float]:
    """
    a trivial algorithm for filling the knapsack
    it takes items in-order from the initial items list (by their appearance in the input file)
    until the knapsack is full
    """
    return add_items_from_list_by_order_until_full(items, capacity)


def greedy_with_densities(items: list, capacity: int) -> [float, float]:
    """
    a GREEDY algorithm for filling the knapsack
    it sorts all items by decreasing densities (value/weight ratio)
    and then adds the items from this density list in-order until the knapsack is full
    """
    sorted_items_by_density = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    return add_items_from_list_by_order_until_full(sorted_items_by_density, capacity)


def dp(items: list, capacity: int) -> [float, float]:
    """a Dynamic Programming approach for filling the knapsack"""
    n = len(items)
    table = [[0] * (n + 1) for _ in range(capacity + 1)]
    taken = [0] * len(items)

    for j in range(1, n + 1):
        item = items[j - 1]

        for i in range(min([i.weight for i in items]), capacity + 1):
            if item.weight <= i:
                table[i][j] = max(
                    table[i][j - 1],  # not selected
                    item.value + table[i - item.weight][j - 1]  # selected
                )
            else:
                table[i][j] = table[i][j - 1]

    value = table[-1][-1]

    # trace back
    j = n
    i = capacity

    while (j > 0) & (i > 0):
        if table[i][j] > table[i][j - 1]:
            taken[j - 1] = 1
            i = i - [x.weight for x in items if x.index == j - 1][0]

        j = j - 1

    return value, taken


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # value, taken = naive(items, capacity)  # naive approach

    if item_count in [400, 10000]:
        value, taken = greedy_with_densities(items, capacity)  # greedy approach
    else:
        value, taken = dp(items, capacity)  # dynamic programming approach

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

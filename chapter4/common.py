#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

ROWS = 4
COLS = 4
GAMMA = 1.0
REWARD = -1.0
THRESHOLD = 0.001

ACTION_MAP = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
}


class Point:

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __add__(self, other):
        return Point(self.row + other.row, self.col + other.col)


TERMINAL_0 = Point(0, 0)
TERMINAL_1 = Point(3, 3)
ACTIONS = {
    0: Point(-1, 0),  # up
    1: Point(0, 1),  # right
    2: Point(1, 0),  # down
    3: Point(0, -1),  # left
}


def is_terminal_state(state):
    return (state == TERMINAL_0) or (state == TERMINAL_1)


def is_on_grid(state):
    r, c = state.row, state.col
    return (0 <= r <= 3) and (0 <= c <= 3)


def initialize_policy(rows, cols):
    size = 4
    prob = 1.0 / size
    policy = {}
    for state in state_generator(rows, cols):
        policy[state] = np.array([prob] * size)
    return policy


def initialize_value_function(rows, cols):
    v = {}
    for state in state_generator(rows, cols):
        v[state] = 0.0
    return v


def state_generator(rows, cols):
    for row in range(rows):
        for col in range(cols):
            yield Point(row, col)


def display_results(v, policy, rows, cols):
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        indices = np.where(policy[state] > 0)
        directions = [ACTION_MAP[index] for index in indices[0]]
        path = ':'.join(directions)
        print(state.row, state.col, path, policy[state])
    print(v)


def display_policy(policy):
    for state in state_generator(ROWS, COLS):
        if is_terminal_state(state):
            continue
        indices = np.where(policy[state] > 0)
        directions = [ACTION_MAP[index] for index in indices[0]]
        path = ':'.join(directions)
        print(state.row, state.col, path)


def display_value_function(v):
    d = np.empty((4, 4))
    for state in state_generator(ROWS, COLS):
        r, c = state.row, state.col
        d[r, c] = v[state]
    print(d)


def overwrite_policy(state, new_policy, policy):
    prob = 1.0 / len(new_policy)
    policy[state] = np.zeros((4,))
    for k in new_policy:
        policy[state][k] = prob

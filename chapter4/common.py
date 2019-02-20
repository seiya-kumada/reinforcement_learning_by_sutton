#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

ROWS = 4
COLS = 4
GAMMA = 1.0
REWARD = -1.0
THRESHOLD = 0.001
ACTIONS = {
    0: np.array([-1, 0]),  # up
    1: np.array([0, 1]),  # right
    2: np.array([1, 0]),  # down
    3: np.array([0, -1]),  # left
}

ACTION_MAP = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
}

TERMINAL_0 = np.array([0, 0])
TERMINAL_1 = np.array([3, 3])


def is_terminal_state(state):
    return (np.all(state == TERMINAL_0)) or (np.all(state == TERMINAL_1))


def is_on_grid(state):
    r, c = state
    return (0 <= r <= 3) and (0 <= c <= 3)


def initialize_policy(rows, cols):
    policy = np.zeros((rows, cols, 4))
    for state in state_generator(rows, cols):
        r, c = state
        policy[r, c] = np.array([0.25, 0.25, 0.25, 0.25])
    return policy


def state_generator(rows, cols):
    for row in range(rows):
        for col in range(cols):
            yield np.array([row, col])

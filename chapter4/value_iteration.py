#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from common import *  # noqa

# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction


def is_terminal_state(state):
    return (np.all(state == TERMINAL_0)) or (np.all(state == TERMINAL_1))


def is_on_grid(state):
    r, c = state
    return (0 <= r <= 3) and (0 <= c <= 3)


def update(v, state, gamma, reward):
    results = {}
    for key in ACTIONS:
        next_state = state + ACTIONS[key]
        if not is_on_grid(next_state):
            next_state = state
        results[key] = reward + gamma * v[next_state[0], next_state[1]]
    max_vs = max(results.values())
    return max_vs


def make_optimal_policy(state, v, reward, gamma):
    results = {}
    for key in ACTIONS:
        next_state = state + ACTIONS[key]
        if not is_on_grid(next_state):
            next_state = state
        results[key] = reward + gamma * v[next_state[0], next_state[1]]
    max_vs = max(results.values())
    return [k for k, val in results.items() if val == max_vs]


def make_policy(v, rows, cols, reward, gamma):
    policy = np.zeros((rows, cols, 4))
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        op = make_optimal_policy(state, v, reward, gamma)
        overwrite_policy(state, op, policy)
    return policy


def value_iteration():
    v = np.zeros((ROWS, COLS))
    k = 0
    while True:
        delta = 0
        for state in state_generator(ROWS, COLS):
            if is_terminal_state(state):
                continue
            r, c = state
            tmp = v[r, c]
            v[r, c] = update(v, state, GAMMA, REWARD)
            delta = max(delta, abs(tmp - v[r, c]))
        k += 1
        if delta < THRESHOLD:
            break
    policy = make_policy(v, ROWS, COLS, REWARD, GAMMA)
    display_results(v, policy, ROWS, COLS)


if __name__ == "__main__":
    value_iteration()

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

    updated_v = 0.0
    for action in ACTIONS.values():
        next_state = state + action
        if not is_on_grid(next_state):
            next_state = state
        updated_v += 0.25 * (reward + gamma * v[next_state[0], next_state[1]])
    return updated_v


def iterative_policy_evaluation():
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

    print(k)
    print(v)


if __name__ == "__main__":
    iterative_policy_evaluation()

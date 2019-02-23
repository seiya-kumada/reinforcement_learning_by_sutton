#!/usr/bin/env python
# -*- coding: utf-8 -*-
from common import *  # noqa
import policy_iteration


def update(v, state, gamma, reward):
    results = {}
    for key in ACTIONS:
        next_state = state + ACTIONS[key]
        if not is_on_grid(next_state):
            next_state = state
        results[key] = reward + gamma * v[next_state]
    max_vs = max(results.values())
    return max_vs


def make_policy(v, rows, cols, reward, gamma):
    policy = initialize_policy(rows, cols)
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        op = policy_iteration.update_policy(state, reward, gamma, v)
        overwrite_policy(state, op, policy)
    return policy


def main():
    v = initialize_value_function(ROWS, COLS)
    while True:
        delta = 0
        for state in state_generator(ROWS, COLS):
            if is_terminal_state(state):
                continue
            tmp = v[state]
            v[state] = update(v, state, GAMMA, REWARD)
            delta = max(delta, abs(tmp - v[state]))
        if delta < THRESHOLD:
            break
    policy = make_policy(v, ROWS, COLS, REWARD, GAMMA)
    display_policy(policy)
    display_value_function(v)


if __name__ == "__main__":
    main()

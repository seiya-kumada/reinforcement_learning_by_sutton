#!/usr/bin/env python
# -*- coding: utf-8 -*-
from common import *  # noqa


def update(v, state, gamma, reward):
    updated_v = 0.0
    for action in ACTIONS.values():
        next_state = state + action
        if not is_on_grid(next_state):
            next_state = state
        updated_v += 0.25 * (reward + gamma * v[next_state])
    return updated_v


def main():
    v = initialize_value_function(ROWS, COLS)
    k = 0
    while True:
        delta = 0
        for state in state_generator(ROWS, COLS):
            if is_terminal_state(state):
                continue
            tmp = v[state]
            v[state] = update(v, state, GAMMA, REWARD)
            delta = max(delta, abs(tmp - v[state]))
        k += 1
        if delta < THRESHOLD:
            break

    print("iteration size: {}".format(k))
    display_value_function(v)


if __name__ == "__main__":
    main()

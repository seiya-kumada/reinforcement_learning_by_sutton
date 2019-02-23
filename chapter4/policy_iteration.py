#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from common import *  # noqa
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction


def evaluate_policy(rows, cols, v, policy, gamma, reward, threshold):
    while True:
        delta = 0
        for state in state_generator(rows, cols):
            if is_terminal_state(state):
                continue
            tmp = v[state]
            v[state] = update_value_function(v, policy, state, gamma, reward)
            delta = max(delta, abs(tmp - v[state]))
        if delta < threshold:
            break


def update_value_function(v, policy, state, gamma, reward):
    action_prob = policy[state]
    updated_v = 0
    for action_key, prob in zip(ACTIONS.keys(), action_prob):
        next_state = state + ACTIONS[action_key]
        if not is_on_grid(next_state):
            next_state = state
        updated_v += prob * (reward + gamma * v[next_state])
    return updated_v


def update_policy(state, policy, reward, gamma, v):
    results = {}
    for key in ACTIONS:
        next_state = ACTIONS[key] + state
        if not is_on_grid(next_state):
            next_state = state
        vs = reward + gamma * v[next_state]
        results[key] = vs
    max_vs = max(results.values())
    return [k for k, val in results.items() if val == max_vs]


def improve_policy(rows, cols, policy, reward, gamma, v):
    is_stable = True
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        old_action_prob = policy[state].copy()
        new_policy = update_policy(state, policy, reward, gamma, v)
        overwrite_policy(state, new_policy, policy)
        if not np.all(old_action_prob == policy[state]):
            is_stable = False
    return is_stable


def main():
    v = initialize_value_function(ROWS, COLS)
    policy = initialize_policy(ROWS, COLS)
    while True:
        evaluate_policy(ROWS, COLS, v, policy, GAMMA, REWARD, THRESHOLD)
        is_stable = improve_policy(ROWS, COLS, policy, REWARD, GAMMA, v)
        if is_stable:
            break
    display_policy(policy)
    display_value_function(v)


if __name__ == "__main__":
    main()

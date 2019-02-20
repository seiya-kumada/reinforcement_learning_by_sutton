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
            r, c = state
            tmp = v[r, c]
            v[r, c] = update(v, policy, state, gamma, reward)
            delta = max(delta, abs(tmp - v[r, c]))
        if delta < threshold:
            break


def update(v, policy, state, gamma, reward):
    r, c = state
    action_prob = policy[r, c]
    updated_v = 0
    for action_key, prob in zip(ACTIONS.keys(), action_prob):
        next_state = state + ACTIONS[action_key]
        if not is_on_grid(next_state):
            next_state = state
        updated_v += prob * (reward + gamma * v[next_state[0], next_state[1]])
    return updated_v


def update_policy(state, policy, reward, gamma, v):
    results = {}
    r, c = state
    for key in ACTIONS:
        next_state = ACTIONS[key] + state
        if not is_on_grid(next_state):
            next_state = state

        vs = reward + gamma * v[next_state[0], next_state[1]]
        results[key] = vs
    max_vs = max(results.values())
    return [k for k, val in results.items() if val == max_vs]


def improve_policy(rows, cols, policy, reward, gamma, v):
    is_stable = True
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        r, c = state
        old_action_prob = policy[r, c].copy()
        new_policy = update_policy(state, policy, reward, gamma, v)

        prob = 1.0 / len(new_policy)
        policy[r, c] = np.zeros((4,))
        for k in new_policy:
            policy[r, c][k] = prob
        if not np.all(old_action_prob == policy[r, c]):
            is_stable = False
    return is_stable


def display_results(v, policy, rows, cols):
    for state in state_generator(rows, cols):
        if is_terminal_state(state):
            continue
        r, c = state
        indices = np.where(policy[r, c] > 0)
        directions = [ACTION_MAP[index] for index in indices[0]]
        path = ':'.join(directions)
        print(r, c, path, policy[r, c])
    print(v)


def policy_iteration():
    v = np.zeros((ROWS, COLS))
    policy = initialize_policy(ROWS, COLS)
    while True:
        evaluate_policy(ROWS, COLS, v, policy, GAMMA, REWARD, THRESHOLD)
        is_stable = improve_policy(ROWS, COLS, policy, REWARD, GAMMA, v)
        if is_stable:
            break
    display_results(v, policy, ROWS, COLS)


if __name__ == "__main__":
    policy_iteration()

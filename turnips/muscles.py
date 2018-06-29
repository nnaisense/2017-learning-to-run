#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint

INIT_ACTIVATION = 0.05
def comp_activation(activation, action):
    activation = np.clip(activation, 0.01, 1.0)
    action = np.clip(action, 0, 1.0)
    u = action
    T_act = 0.01
    T_deact = 0.04
    Dt = 0.01

    def tau(a, u):
        return np.where(u > a, T_act * (0.5 + 1.5 * a), T_deact / (0.5 + 1.5 * a))

    def dy(a, t):
        return (u - a) / tau(a, u)

    sol = odeint(dy, activation, [0, Dt])
    return sol[1, 0] if sol.shape[1] == 1 else sol[1, :]

if __name__ == '__main__':
    T = 1000
    M = 18
    activations = np.empty(shape=[T, M])
    activations[0, :] = INIT_ACTIVATION
    actions = np.random.rand(T-1, M)

    print('initial activation', activations[0, 0])
    for t in range(1, T):
        activations[t, :] = comp_activation(activations[t - 1, :], actions[t - 1, :])
        print('action', actions[t - 1, 0], 'leads to activation', activations[t, 0])

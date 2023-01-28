import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.patches import Rectangle

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.Variable(tf.zeros([sdim]))

    policy = np.zeros((sdim))
    for pos, s in enumerate(problem["idx2pos"]):
        aS = actions_space(s, adim, problem["m"], problem["n"])
        policy[pos] = np.random.choice(aS)

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for ii in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        err = 0.
        for pos, s in enumerate(problem["idx2pos"]):
            aS = actions_space(s, adim,  problem["m"],  problem["n"])
            v = bellman_optimality_update(problem, V, s, pos, aS, adim, gam, reward)
            if terminal_mask[pos] != 1:
                best = max(v)
                policy[pos] = np.argmax(v)
            else:
                temp = []
                for u in aS:
                    temp.append(reward[pos, u])
                best = max(temp)

            err = max(err, tf.linalg.norm(V[pos] - best))
            V[pos].assign(best)

        if ii % 2 == 0:
            print(tf.get_static_value(err))

        if tf.get_static_value(err) < 1e-7:
            print(ii)
            print(tf.get_static_value(err))
            break
        ######### Your code ends here ###########'
    return V, policy

## Compute the bellman equation presented in Problem 01
def bellman_optimality_update(problem, V, s, s_pos, aS, adim, gamma, reward):
    values = np.zeros((adim))
    for u in aS:
        for u_iter in aS:
            if u == u_iter:
                s_next = next_state(problem["m"], problem["n"], s, u)
            else:
                s_next = next_state(problem["m"], problem["n"], s, u_iter)

            idx_next = problem["pos2idx"][s_next[0], s_next[1]]
            p = problem["Ts"][u][s_pos, idx_next]
            values[u] += tf.convert_to_tensor(gamma * p * V[idx_next])

        values[u] += reward[s_pos, u]

    return values

## Compute the permitted action for each state
def actions_space(state, adim, m, n):
    act = []
    for u in range(adim):
        s_next = next_state(m, n, state, u)
        if state[0] != s_next[0] or state[1] != s_next[1]:
            act.append(u)
    return act

## Function to plot heatmap from scratch
def plot_heatmap(problem, data, m, n, ant):
    dt = np.reshape(np.array(data), (m,n)).T
    ax = sb.heatmap(np.round(dt, 3), annot=ant)
    ax.invert_yaxis()
    p = path(problem, data, [0, 0], [19, 9], m, n)
    for t in p:
        ax.add_patch(Rectangle(t, 1, 1, fill=False, edgecolor='blue', lw=3))
    plt.show()

## Function to simulate the MDP starting from x = (0, 0) over N = 100 time steps
def path(problem, policy, s_start, s_goal, m, n):
    p = []
    s = s_start
    N = 100
    p.append(s_start)
    iteration = 0
    while iteration <= N:
        if s[0] == s_goal[0] and s[1] == s_goal[1]:
            break
        s = next_state(m, n, s, policy[problem["pos2idx"][s[0], s[1]]])
        p.append(s)
        iteration += 1
    return p

## Compute the next state s next given an actual state and ction   
def next_state(m, n, s, u):
    xclip = lambda x: min(max(0, x), m - 1)
    yclip = lambda y: min(max(0, y), n - 1)
    # {right, up, left, down}
    if u == 0:
        return np.array([xclip(s[0] + 1), yclip(s[1] + 0)])
    elif u == 1:
        return np.array([xclip(s[0] + 0), yclip(s[1] + 1)])
    elif u == 2:
        return np.array([xclip(s[0] - 1), yclip(s[1] + 0)])
    else:
        return np.array([xclip(s[0] + 0), yclip(s[1] - 1)])

# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt, policy = value_iteration(problem, reward, terminal_mask, gam)

    # plt.figure(213)
    # visualize_value_function(np.array(V_opt).reshape((n, n)))
    # plt.title("value iteration")
    # plt.show()
    plot_heatmap(problem, V_opt, n, n, False)
    plot_heatmap(problem, policy, n, n, True)

if __name__ == "__main__":
    main()

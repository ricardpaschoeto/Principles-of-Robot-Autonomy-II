import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import seaborn as sb

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2
    
    m = problem["m"]
    n = problem["n"]
    #aS = []
    #actions_space(problem["idx2pos"], aS, adim, problem["m"], problem["n"])
    #V = tf.Variable(V)
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

        ######### Your code ends here ###########
        err = 0.
        if ii % 20 == 0:
            print(ii)

        for pos, s in enumerate(problem["idx2pos"]):
            aS = actions_space(s, adim, problem["m"], problem["n"])
            best = bellman_optimality_update(problem, V, s, pos, aS, adim, gam, reward, terminal_mask)
            err = max(err, tf.linalg.norm(V[pos] - best))
            tf.Variable(V[pos]).assign(best)

        if ii % 20 == 0:
            print(tf.get_static_value(err))

        if tf.get_static_value(err) < 1e-7:
            print(ii)
            print(tf.get_static_value(err))
            break

    return V

def bellman_optimality_update(problem, V, s, s_pos, aS, adim, gamma, reward, terminal_mask):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    values = np.zeros((adim))
    if terminal_mask[s_pos] != 1:
        for u in aS[s_pos]:
            transition = problem["Ts"][u]
            s_next = next_state(problem["m"], problem["n"], s, u)
            idx = problem["pos2idx"][s_next[0], s_next[1]]
            p = transition[problem["pos2idx"][s[0], s[1]], idx]            
            values[u] += gamma * p * V[idx]

            best = np.max(reward[s_pos, u] + values[u])   
    else:
        best = reward[s_pos, 0]

    return best

def actions_space(state, adim, m, n):

    act = []
    for u in range(adim):
        s_next = next_state(m, n, state, u)
        if state[0] != s_next[0] or state[1] != s_next[1]:
            act.append(u)
    return act
    
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

def plot_heatmap(data, ant):
    m = data.T
    ax = sb.heatmap(np.round(m, 3), annot=ant)
    ax.invert_yaxis()
    plt.show()

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
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()

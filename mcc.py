from easy21 import Environment, State
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import pandas as pd
import time


def mcc():
    """Monte Carlo Control algorithm
    """
    n0 = 100
    env = Environment()
    tic = time.time()
    for i in range(0, 500000):
        state = State()
        state_lst = [state]
        state_action = [[state, state.policy]]
        g = 0
        while True:
            step = env.step(state_lst[-1], state_lst[-1].policy)
            state_lst.append(step[0])
            g += step[1]
            if step[0].terminal:
                break
            state_action.append([step[0], step[0].policy])
        env.inc_ns(state_lst)
        env.inc_nsa(state_action)
        env.inc_q(state_action, reward=g)
        env.inc_policy(state_action, n0)

    toc = time.time()
    player = [int(x.split(',')[0].split(' ')[1]) for x in list(env.q.keys())]
    dealer = [int(x.split(',')[1].split(' ')[2]) for x in list(env.q.keys())]
    df = pd.DataFrame(env.q.values())
    df['player'] = player
    df['dealer'] = dealer
    df['optimal'] = env.optimal_q()
    X = df[df['player'] > 11]['player']
    Y = df[df['player'] > 11]['dealer']
    Z = df[df['player'] > 11]['optimal']
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='twilight')
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('maxQ*(s,a)')
    ax.set_title(f'Monte Carlo Control after 500,000 Episodes in {toc-tic:.2f} seconds')
    plt.show()


if __name__ == '__main__':
    print(mcc())

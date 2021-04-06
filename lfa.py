from easy21 import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sarsa import *
from mcc import *


def lfa():
    tic = time.time()
    mse = []
    mse_0 = []
    mse_1 = []
    params = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    mcc = mcc2(1000)

    for p in params:
        tic2 = time.time()
        env = Environment()
        for i in range(0, 1000):
            env.lfa_e = np.zeros((2, 6, 3))
            g = 0
            state = State()
            state_action = [[state, env.get_action(state)]]
            while True:
                phi_ = np.zeros((2, 6, 3))
                step = env.step(state_action[-1][0], state_action[-1][1])
                g += step[1]
                phi = env.get_feature(state_action[-1])
                td_error = env.lfa_td_error(state_action[-1], reward=g, new_state=step[0])
                env.inc_lfa_e(state_action[-1])
                env.inc_lfa_q(phi)
                env.inc_lfa_policy()
                if p == 0:
                    env.lfa_e[phi[0], phi[1], phi[2]] = 1
                else:
                    env.lfa_e *= p
                env.inc_w(td_error)
                #print(step[0].sample)
                if step[0].terminal:
                    break
                state_action.append([step[0], env.get_action(step[0])])
            if p == 0:
                mse_0.append(env.get_lfa_mse(mcc=mcc))
            if p == 1:
                mse_1.append(env.get_lfa_mse(mcc=mcc))
        mse.append(env.get_lfa_mse(mcc=mcc))
        toc2 = time.time()
        print(f'{toc2 - tic2:.2f} seconds for {p}')
    toc = time.time()
    print(f'{toc - tic:2f} seconds taken')

    sns.set_style('darkgrid')
    fig, axes = plt.subplots()
    sns.lineplot(x=params, y=mse, ax=axes)
    axes.set(xlabel='Lambda', ylabel='Mean-Squared Error',
             title='MSE of Sarsa(lambda) Linear Function Approximation')
    plt.show()

    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    sns.lineplot(x=np.arange(0, 1000), y=mse_0, ax=ax[0])
    ax[0].set(xlabel='Episodes', ylabel='Mean-Squared Error',
              title='Lambda = 0,1 Learning Curves Linear Function Approximation')
    ax[0].legend(['Lambda = 0'])
    sns.lineplot(x=np.arange(0, 1000), y=mse_1, ax=ax[1])
    ax[1].set(xlabel='Episodes', ylabel='Mean-Squared Error')
    ax[1].legend(['Lambda = 1'])
    plt.show()


if __name__ == '__main__':
    print(lfa())

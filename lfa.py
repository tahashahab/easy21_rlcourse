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
        for i in range(0, 100):
            g = 0
            state = State()
            state_lst = [state]
            if state.sample not in env.q.keys():
                state_action = [[state, state.policy]]
            else:
                state_action = [[state, env.q[state.sample]['policy']]]
            while True:
                if state_lst[-1].sample not in env.q.keys():
                    step = env.step(state_lst[-1], state_lst[-1].policy)
                else:
                    step = env.step(state_lst[-1], env.q[state_lst[-1].sample]['policy'])
                g += step[1]
                td_error = env.td_error(state_action[-1], reward=g, new_state=step[0])
                env.inc_e_sa(state_action[-1])
                env.inc_lfa_q(state_action[-1], tde=td_error)
                env.inc_lfa_policy()
                env.inc_e(param=p)
                if step[0].terminal:
                    break
                state_lst.append(step[0])
                if step[0].sample in env.q.keys():
                    state_action.append(
                        [step[0], env.q[step[0].sample]['policy']])
                else:
                    state_action.append([step[0], step[0].sample])
            if p == 0:
                mse_0.append(get_mse(env.q, mcc))
            if p == 1:
                mse_1.append(get_mse(env.q, mcc))
        mse.append(get_mse(env.q, mcc))
        toc2 = time.time()
        print(f'{toc2 - tic2:.2f} seconds for {p}')
    toc = time.time()
    print(f'{toc - tic:2f} seconds taken')

    sns.set_style('darkgrid')
    fig, axes = plt.subplots()
    sns.lineplot(x=params, y=mse, ax=axes)
    axes.set(xlabel='Lambda', ylabel='Mean-Squared Error',
             title='MSE of each Lambda')
    plt.show()

    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    sns.lineplot(x=np.arange(0, 100), y=mse_0, ax=ax[0])
    ax[0].set(xlabel='Episodes', ylabel='Mean-Squared Error',
              title='Lambda = 0,1 Learning Curves')
    ax[0].legend(['Lambda = 0'])
    sns.lineplot(x=np.arange(0, 100), y=mse_1, ax=ax[1])
    ax[1].set(xlabel='Episodes', ylabel='Mean-Squared Error')
    ax[1].legend(['Lambda = 1'])
    plt.show()


if __name__ == '__main__':
    pass

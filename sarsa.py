from easy21 import State, Environment
import time

def sarsa():
    """Sarsa(lambda) algorithm
    """
    n0 = 100
    env = Environment()
    tic = time.time()
    mse = []

    for i in range(0, 11000):
        param = env.get_lambda(i)
        state = State()
        state_lst = [state]
        state_action = [[state, state.policy]]
        g = 0
        while True:
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
                env.inc_sarsa_q(state_action[-1], reward=g, param=param)
                env.inc_sarsa_policy(state_action[-1], n0)


    toc = time.time()


if __name__ == '__main__':
    print(sarsa())

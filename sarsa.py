from easy21 import State, Environment
import time

def sarsa():
    """Sarsa(lambda) algorithm
    """
    n0 = 100
    env = Environment()
    tic = time.time()
    mse = []
    params = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for p in params:
        for i in range(0, 1000):
            g = 0
            state = State()
            state_lst = [state]
            state_action = [[state, state.policy]]
            while True:
                step = env.step(state_lst[-1], state_lst[-1].policy)
                g += step[1]
                env.inc_sarsa_ns(state_lst[-1])
                env.inc_sarsa_nsa(state_action[-1])
                td_error = env.td_error(state_action, reward=g, new_state=step[0])
                env.inc_e_sa(state_action[-1])
                env.inc_sarsa_q()
                env.inc_sarsa_policy(state_action[-1], n0)
                env.inc_e(param=p)
                if step[0].is_terminal():
                    break
                state_action.append([step[0], step[0].policy])
    toc = time.time()


if __name__ == '__main__':
    print(sarsa())

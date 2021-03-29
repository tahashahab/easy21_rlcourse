import random
import numpy as np


class TerminalStateError(BaseException):
    pass


def draw():
    if random.randint(1, 3) == 1:
        card = -random.randint(1, 10)
    else:
        card = random.randint(1, 10)
    return card


class State:

    def __init__(self, player=None, dealer=None) -> None:

        if player is None:
            self.player = random.randint(1, 10)
        else:
            self.player = player
        if dealer is None:
            self.dealer = random.randint(1, 10)
        else:
            self.dealer = dealer

        self.sample = f'Player: {self.player}, Dealer: {self.dealer}'
        self.terminal = False
        self.ns = 0
        self.nsa = {'hit': 0, 'stick': 0}
        self.policy = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])

    def decision(self):
        if self.dealer > self.player:
            return -1
        elif self.dealer < self.player:
            return 1
        else:
            return 0

    def is_terminal(self):
        self.terminal = True

    def state_action_pair(self, action):
        return f'Player sum: {self.player}, Dealer sum: {self.dealer}, {action}'


class Environment:

    def __init__(self) -> None:
        self.state = None
        self.ns = {}
        self.nsa = {}
        self.state_action = {}
        self.q = {}
        self.e = {}

    def step(self, state: State, action):
        assert action in ['hit', 'stick'], 'Actions limited to hit or stick'
        if state.terminal:
            raise TerminalStateError
        elif action == 'hit':
            state.player += draw()
            if state.player < 1 or state.player > 21:
                new_player = state.player
                new_dealer = state.dealer
                next_state = State(player=new_player, dealer=new_dealer)
                reward = -1
                next_state.is_terminal()
                return next_state, reward, 'Terminal'
            else:
                new_player = state.player
                new_dealer = state.dealer
                next_state = State(player=new_player, dealer=new_dealer)
                reward = 0
                return next_state, reward, 'Non-Terminal'
        else:
            while 0 < state.dealer <= 17:
                state.dealer += draw()
            if state.dealer < 1 or state.dealer > 21:
                new_player = state.player
                new_dealer = state.dealer
                next_state = State(player=new_player, dealer=new_dealer)
                reward = 1
                next_state.is_terminal()
                return next_state, reward, 'Terminal'
            else:
                new_player = state.player
                new_dealer = state.dealer
                next_state = State(player=new_player, dealer=new_dealer)
                reward = next_state.decision()
                next_state.is_terminal()
                return next_state, reward, 'Terminal'

    def inc_ns(self, state_lst: list):
        for s in state_lst:
            if s not in self.ns.keys():
                self.ns[s] = 1
            else:
                self.ns[s] += 1

    def inc_nsa(self, state_action: list):
        for s in state_action:
            if s[0].sample not in self.nsa.keys():
                if s[1] == 'hit':
                    self.nsa[s[0].sample] = {'hit': 1, 'stick': 0}
                else:
                    self.nsa[s[0].sample] = {'hit': 0, 'stick': 1}
            else:
                if s[1] == 'hit':
                    self.nsa[s[0].sample]['hit'] += 1
                else:
                    self.nsa[s[0].sample]['stick'] += 1

    def inc_q(self, state_action: list, reward):
        for s in state_action:
            if s[0].sample not in self.q.keys():
                self.q[s[0].sample] = {'hit': 0, 'stick': 0}
                self.q[s[0].sample][s[1]] = reward
            else:
                self.q[s[0].sample][s[1]] += (1 / self.nsa[s[0].sample][s[1]]) * (reward - self.q[s[0].sample][s[1]])

    def optimal_q(self):
        op = []
        for d in self.q.values():
            op.append(np.max(list(d.values())))
        return op


    def inc_policy(self, state_action, n0):
        for s in state_action:
            epsilon = (n0 / (n0 + self.ns[s[0]]))

            if self.q[s[0].sample]['hit'] > self.q[s[0].sample]['stick']:
                greedy_action = 'hit'
            elif self.q[s[0].sample]['hit'] < self.q[s[0].sample]['stick']:
                greedy_action = 'stick'
            else:
                greedy_action = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])

            epsilon_action = np.random.choice(['random', 'greedy'], p=[epsilon, 1 - epsilon])
            if epsilon_action == 'random':
                s[0].policy = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])
            else:
                s[0].policy = greedy_action

    def inc_sarsa_ns(self, state: State):
            if State not in self.ns.keys():
                self.ns[state.sample] = 1
            else:
                self.ns[state.sample] += 1

    def inc_sarsa_nsa(self, state_action: list):
        if state_action[0].sample not in self.nsa.keys():
            if state_action[1] == 'hit':
                self.nsa[state_action[0].sample] = {'hit': 1, 'stick': 0}
            else:
                self.nsa[state_action[0].sample] = {'hit': 0, 'stick': 1}
        else:
            if state_action[1] == 'hit':
                self.nsa[state_action[0].sample]['hit'] += 1
            else:
                self.nsa[state_action[0].sample]['stick'] += 1

    def inc_sarsa_q(self, sa: list, tde):
        if sa[0].sample not in self.q.keys():
            self.q[sa[0].sample] = {'hit': 0, 'stick': 0}
        for key in self.q.keys():
            self.q[key]['hit'] += (1/self.nsa[key]['hit']) * tde * (self.e[key]['hit'])
            self.q[key]['stick'] += (1 / self.nsa[key]['stick']) * tde * (self.e[key]['stick'])
#TODO fix sara policy so that it takes a state value to loop with so i can change the policies of the states
    def inc_sarsa_policy(self, sa: list, n0):
        for key in self.q.keys():
            epsilon = (n0 / (n0 + self.ns[key]))

            if self.q[key]['hit'] > self.q[key]['stick']:
                greedy_action = 'hit'
            elif self.q[key]['hit'] < self.q[key]['stick']:
                greedy_action = 'stick'
            else:
                greedy_action = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])

            epsilon_action = np.random.choice(['random', 'greedy'], p=[epsilon, 1 - epsilon])

            if epsilon_action == 'random':
                sa[0].policy = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])
            else:
                sa[0].policy = greedy_action

    def inc_e_sa(self, state_action: list):
        if state_action[0].sample not in self.e.keys():
            self.e[state_action[0].sample] = {'hit': 0, 'stick': 0}
        else:
            if state_action[1] == 'hit':
                self.e[state_action[0].sample]['hit'] = self.e[state_action[0].sample]['hit'] + 1
            else:
                self.e[state_action[0].sample]['stick'] = self.e[state_action[0].sample]['stick'] + 1

    def inc_e(self, param):
        for key in self.e.keys():
            self.e[key]['hit'] = param*self.e[key]['hit']
            self.e[key]['stick'] = param*self.e[key]['stick']

    def td_error(self, state_action: list, reward, new_state: State):
        qsa = 0
        qsa_prime = 0
        if state_action[0].sample in self.q.keys():
            qsa = self.q[state_action[0].sample][state_action[1]]
        if new_state.sample in self.q.keys():
            qsa_prime = self.q[new_state.sample][new_state.policy]
        return reward + qsa_prime + qsa


if __name__ == '__main__':
    pass

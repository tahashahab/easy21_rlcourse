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
        self.features = {'dealer': [[1, 4], [4, 7], [7, 10]],
                         'player': [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18],
                                    [16, 21]],
                         'action': ['hit', 'stick']}
        self.w = np.zeros((2, 6, 3))
        self.lfa_e = np.zeros((2, 6, 3))
        self.lfa_q = np.zeros((2, 6, 3))
        self.policy = np.zeros((6, 3))

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
            self.q[sa[0].sample] = {'hit': 0, 'stick': 0, 'policy': sa[1]}
        for key in self.q.keys():
            if key not in self.e.keys():
                continue
            else:
                if self.nsa[key]['hit'] > 0:
                    self.q[key]['hit'] += (1/self.nsa[key]['hit']) * tde * (self.e[key]['hit'])
                if self.nsa[key]['stick'] > 0:
                    self.q[key]['stick'] += (1/self.nsa[key]['stick']) * tde * (self.e[key]['stick'])

    def inc_sarsa_policy(self, n0):
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
                self.q[key]['policy'] = np.random.choice(['hit', 'stick'], p=[0.5, 0.5])
            else:
                self.q[key]['policy'] = greedy_action

    def inc_e_sa(self, state_action: list):
        if state_action[0].sample not in self.e.keys():
            self.e[state_action[0].sample] = {'hit': 0, 'stick': 0}
        else:
            if state_action[1] == 'hit':
                self.e[state_action[0].sample]['hit'] += 1
            else:
                self.e[state_action[0].sample]['stick'] += 1

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
            qsa_prime = self.q[new_state.sample][self.q[new_state.sample]['policy']]
        return reward + qsa_prime - qsa

    def inc_lfa_q(self, phi: list):
        self.lfa_q[phi[0], phi[1], phi[2]] = self.w[phi[0], phi[1], phi[2]]

    def inc_lfa_policy(self):
        epsilon = 0.05

        for dealer_index, player_list in enumerate(self.policy):
            for player_index, policy_value in enumerate(player_list):
                if self.lfa_q[0, dealer_index, player_index] > self.lfa_q[1, dealer_index, player_index]:
                    greedy_action = 0
                elif self.lfa_q[0, dealer_index, player_index] < self.lfa_q[1, dealer_index, player_index]:
                    greedy_action = 1
                else:
                    greedy_action = np.random.choice([0, 1], p=[0.5, 0.5])

                epsilon_action = np.random.choice(['random', 'greedy'], p=[epsilon, 1 - epsilon])
                if epsilon_action == 'random':
                    self.policy[dealer_index, player_index] = np.random.choice([0, 1], p=[0.5, 0.5])
                else:
                    self.policy[dealer_index, player_index] = greedy_action

    def lfa_td_error(self, state_action: list, reward, new_state: State):
        phi = self.get_feature(state_action)
        phi_prime = self.get_feature([new_state, self.get_action(new_state)])
        qsa = self.lfa_q[phi[0], phi[1], phi[2]]
        qsa_prime = self.lfa_q[phi_prime[0], phi_prime[1], phi_prime[2]]
        return reward + qsa_prime - qsa

    def get_feature(self, sa: list):
        if sa[1] == 'hit':
            action_index = 0
        else:
            action_index = 1
        phi = np.zeros((2, 6, 3))
        for dealer_index, dealer_list in enumerate(self.features['dealer']):
            if dealer_list[0] <= sa[0].dealer <= dealer_list[1]:
                for player_index, player_list in enumerate(self.features['player']):
                    if player_list[0] <= sa[0].player <= player_list[1]:
                        phi = [action_index, dealer_index, player_index]
        return phi

    def inc_lfa_e(self, sa: list):
        if sa[1] == 'hit':
            action_index = 0
        else:
            action_index = 1
        for dealer_index, dealer_list in enumerate(self.features['dealer']):
            if dealer_list[0] <= sa[0].dealer <= dealer_list[1]:
                for player_index, player_list in enumerate(self.features['player']):
                    if player_list[0] <= sa[0].player <= player_list[1]:
                        self.lfa_e[action_index, dealer_index, player_index] += 1

    def inc_w(self, td_error):
        self.w = 0.01 * td_error * self.lfa_e

    def get_action(self, state: State):
        for dealer_index, dealer_list in enumerate(self.features['dealer']):
            if dealer_list[0] <= state.dealer <= dealer_list[1]:
                for player_index, player_list in enumerate(self.features['player']):
                    if player_list[0] <= state.player <= player_list[1]:
                        if self.policy[dealer_index, player_index] == 0:
                            action = 'hit'
                        else:
                            action = 'stick'
        return action

    def get_lfa_mse(self, mcc):
        mse = 0
        count = 0
        for key in mcc.keys():
            player = int(key.split(',')[0].split(' ')[1])
            dealer = int(key.split(',')[1].split(' ')[2])
            for dealer_index, dealer_list in enumerate(self.features['dealer']):
                if dealer_list[0] <= dealer <= dealer_list[1]:
                    for player_index, player_list in enumerate(self.features['player']):
                        if player_list[0] <= player <= player_list[1]:
                            phi = [dealer_index, player_index]

            mse += (self.lfa_q[0, phi[0], phi[1]] - mcc[key]['hit']) ** 2
            mse += (self.lfa_q[1, phi[0], phi[1]] - mcc[key]['stick']) ** 2
            count += 2
        return mse / count


if __name__ == '__main__':
    pass

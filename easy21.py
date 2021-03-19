import random


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

        self.sample = f'Player sum: {self.player}, Dealer sum: {self.dealer}'
        self.terminal = False
        self.ns = 0
        self.nsa = {'hit': 0, 'stick': 0}

    def decision(self):
        if self.dealer > self.player:
            return -1
        elif self.dealer < self.player:
            return 1
        else:
            return 0

    def is_terminal(self):
        self.terminal = True


class Environment:

    def __init__(self) -> None:
        self.state = None
        self.reward = 0

    def step(self, state, action):
        assert action in ['hit', 'stick'], 'Actions limited to hit or stick'
        state.ns += 1
        state.nsa[action] += 1
        if state.terminal:
            raise TerminalStateError
        elif action == 'hit':
            state.player += draw()
            if state.player < 1 or state.player > 21:
                self.state = State(player=state.player, dealer=state.dealer)
                self.reward -= 1
                self.state.is_terminal()
                return self.state.sample, self.reward, 'Terminal'
            else:
                self.state = State(player=state.player, dealer=state.dealer)
                return self.state.sample, self.reward
        else:
            while 1 < state.dealer <= 17:
                state.dealer += draw()
            if state.dealer < 1 or state.dealer > 21:
                self.state = State(player=state.player, dealer=state.dealer)
                self.reward += 1
                self.state.is_terminal()
                return self.state.sample, self.reward, 'Terminal'
            else:
                self.state = State(player=state.player, dealer=state.dealer)
                self.reward += state.decision()
                self.state.is_terminal()
                return self.state.sample, self.reward, 'Terminal'

    def step2(self, state, action):
        assert action in ['hit', 'stick'], 'Actions limited to hit or stick'
        state.ns += 1
        state.nsa[action] += 1
        if state.terminal:
            raise TerminalStateError
        elif action == 'hit':
            state.player += draw()
            if state.player < 1 or state.player > 21:
                next_state = State(player=state.player, dealer=state.dealer)
                self.reward -= 1
                next_state.is_terminal()
                return next_state.sample, self.reward, 'Terminal'
            else:
                next_state = State(player=state.player, dealer=state.dealer)
                return next_state.sample, self.reward
        else:
            while 1 < state.dealer <= 17:
                state.dealer += draw()
            if state.dealer < 1 or state.dealer > 21:
                next_state = State(player=state.player, dealer=state.dealer)
                self.reward += 1
                next_state.is_terminal()
                return next_state.sample, self.reward, 'Terminal'
            else:
                next_state = State(player=state.player, dealer=state.dealer)
                self.reward += state.decision()
                next_state.is_terminal()
                return next_state.sample, self.reward, 'Terminal'

if __name__ == '__main__':
    s = State()
    env = Environment()
    s2 = env.step(s, 'hit')[0]
    print(env.step2(s2, 'hit'))

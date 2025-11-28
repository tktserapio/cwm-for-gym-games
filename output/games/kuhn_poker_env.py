import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        return self.state, {}

    def step(self, action):
        row, col = divmod(action, 3)
        if self.state[row, col] != 0:
            return self.state, -10, False, False, {}

        self.state[row, col] = self.current_player
        winner = self.check_winner()

        if winner is not None:
            reward = 1 if winner == self.current_player else -1
            terminated = True
        elif not self.valid_moves():
            reward = 0
            terminated = True
        else:
            reward = 0
            terminated = False

        self.current_player *= -1
        return self.state, reward, terminated, False, {}

    def valid_moves(self):
        return [i for i in range(9) if self.state[i // 3, i % 3] == 0]

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        board = '\n'.join([' '.join([symbols[self.state[row, col]] for col in range(3)]) for row in range(3)])
        print(board)

    def check_winner(self):
        lines = [
            self.state[0, :], self.state[1, :], self.state[2, :],  # rows
            self.state[:, 0], self.state[:, 1], self.state[:, 2],  # columns
            self.state.diagonal(), np.fliplr(self.state).diagonal()  # diagonals
        ]
        for line in lines:
            if np.all(line == self.current_player):
                return self.current_player
        return None
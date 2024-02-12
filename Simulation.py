import gym
from gym import spaces
import numpy as np
from FireMap import FireMap, make_board, dimensions

class FireMapEnv(gym.Env):
    """
    Wrapper for FireMap environment that follows OpenAI Gymnasium interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FireMapEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete((6,6,6))
        obs_dimensions = dimensions()
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_dimensions, dtype=np.float)
        self.state = None

    def step(self, action):
        """
        Take step based on current action

        Actions:
        0 - Move north
        1 - Move south
        2 - Move east
        3 - Move west
        4 - Add moisture to current cell (decrease intensity/chance of fire)
        5 - Remove fuel from current cell (only on non-active cell)
        """
        self.state.next(action)
        obs = self.state.state # N * N * L np.array
        reward = self.state.get_reward() # Count of active fires * -1
        done = self.state.get_done() # Done when all fires are extinguished
        info = self.state.get_info() # None
        return obs, reward, done, info

    def reset(self):
        """
        Creates new board with a pre-extinguished block to decrease homogeneity
        New board has ~5 starting fires
        """
        board = make_board()
        self.state = FireMap(board)
        return self.state.state

    def render(self, mode='human'):
        """
        Print current state as N * N chart
        """
        self.state.show()

    def close(self):
        """
        No wrap-up methods required
        """
        pass

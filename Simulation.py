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
        obs_dimensions = dimensions()
        self.action_space = spaces.MultiDiscrete((30, 30, 30, 30, 30, 30))
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_dimensions, dtype=np.float)
        self.state = None

    def step(self, action):
        """
        Take step based on current action

        Actions:
        coordinates of all 3 agents
        [(x1, y1), (x2, y2), (x3, y3)]
        """
        self.state.next(action)
        obs = self.state.state
        reward = self.state.get_reward()
        done = self.state.get_done()
        info = self.state.get_info()
        return obs, reward, done, info

    def reset(self):
        """
        Creates new board with a pre-extinguished block to decrease homogeneity
        New board has ~5 starting fires
        """
        board = make_board()
        self.state = FireMap(board)
        return self.state.state

    def render(self, detailed: bool = False, mode: str ='human'):
        """
        Print current state as N * N chart
        """
        if detailed:
            self.state.show_detailed()
        else:
            self.state.show()

    def close(self):
        """
        No wrap-up methods required
        """
        pass

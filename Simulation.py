import gym
import numpy as np
from FireMap import FireMap, make_board, dimensions


class FireMapEnv(gym.Env):
    """
    Wrapper for FireMap environment that follows the OpenAI Gymnasium interface.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(FireMapEnv, self).__init__()
        self.action_space = gym.spaces.MultiDiscrete((30, 30, 30, 30, 30, 30))
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=dimensions(), dtype=np.float
        )
        self.state = None

    def step(self, action: list[tuple[int, int]] = []):
        """
        Take step based on current action

        Actions:
        Coordinates of all 3 agents downsampled by 3
        90 * 90 board, but 30 * 30 action space
        [(x0, y0), (x1, y1), (x2, y2)]
        """
        self.state.next(action)
        obs = self.state.state
        reward = self.state.get_reward()
        done = self.state.get_done()
        info = self.state.get_info()
        return obs, reward, done, info

    def reset(self, seed: int = None):
        """
        Generates a new board with fuel and moisture distributed according to perlin noise.
        Unit wind vector is generated as well.
        Initialises with a few points of fire.
        """
        if not seed:
            seed = np.random.randint(0, 10000)

        board = make_board(seed=seed)
        self.state = FireMap(board, seed=seed)
        return self.state.state

    def render(self, detailed: bool = False, mode: str = "human"):
        """
        --- Detailed View ---
        Fire Intensity Channel: Actively burning cells with intensity in one of 6 classes
        Fuel Channel: Percentage of fuel remaining (initialised with Perlin noise)
        Moisture Channel: Percentage of fuel remaining (initialised with Perlin noise). This is the channel which the agent actively engages with.
        Agent Channel: A visualization of the distribution of next moves for the heuristic agent. This is a combination of distance from nearest active fire, being upwind/downwind of fire, and current moisture in that cell.
            - Agent channel includes wind vector

        --- Standard View ---
        Base layer: Fire intensity (see above)
        Semi-transparent layer: Moisture overlay (darker -> more moist)
        Overlays:
            - Wind vector
            - Agent scatter points (if enabled)
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

import numpy as np
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display

MAP_SIZE = 100
MAX_INTENSITY = 6
START_INTENSITY = 3
STARTING_FIRES = 5

AGENTS = [0,0,0]
# Snap, Crackle, and Pop
# Or Alvin, Simon, and Theodore
INTENSITY, FUEL, MOISTURE, ELEVATION, AGENTS[0], AGENTS[1], AGENTS[2], SELF_EXTINGUISH, NUM_LAYERS = 0, 1, 2, 3, 4, 5, 6, 7, 8
IGNITION_RATE = 0.2
MOVE_TABLE = {
    0: (0, 1),
    1: (0, -1),
    2: (1, 0),
    3: (-1, 0),
    4: (0, 0),
    5: (0, 0),
}

FIRST_JUMP_RATIO = 1.2
STARTING_FUEL = 100
FUEL_VARIANCE = 2


class FireMap:
    """
    State values:
    0: Intensity of fire [0-6]
    1: Fuel available (starts at 100; burn ~intensity each day)
    2: Moisture
    3: Elevation
    4,5,6: Position of agents (one-hot)
    7: Cell has self-extinguished
    8,9: Wind (x,y), will be a scalar across entire map
    10: Humidity, will be a scalar across entire map
    """
    def __init__(self, board) -> None:
        self.state = np.zeros((MAP_SIZE,MAP_SIZE, NUM_LAYERS))
        self.state[:, :, INTENSITY] = board
        self.state[:, :, FUEL] = np.random.randint(STARTING_FUEL / FUEL_VARIANCE, STARTING_FUEL, (MAP_SIZE, MAP_SIZE))
        
        for agent in AGENTS:
            self.state[np.random.randint(10,MAP_SIZE - 10),np.random.randint(10,MAP_SIZE - 10),agent] = 1

        self.time = 0

    def next(self, actions) -> None:
        intensity = self.state[:, :, INTENSITY]
        fuel = self.state[:, :, FUEL]
        moisture = self.state[:, :, MOISTURE]
        elevation = self.state[:, :, ELEVATION]

        self._move_agents(actions)
        
        intensity_of_neighbours, non_zero_neighbours = self._get_neighbours(intensity)
        intensity = self._get_new_ignitions(intensity, intensity_of_neighbours, non_zero_neighbours)
        intensity = np.where(fuel >= 0, intensity, -10)

        self.state[:, :, INTENSITY] = np.where(intensity > MAX_INTENSITY, MAX_INTENSITY, intensity//1)
        self.state[:, :, FUEL] -= self.state[:, :, INTENSITY] # Consume fuel based on intensity
        self.time += 1
        
    def _move_agents(self, actions) -> None:
        """
        Move agents position
        """
        moves = [MOVE_TABLE[action] for action in actions]
        for i in range(3):
            dx, dy = moves[i]
            agent = AGENTS[i]
            x, y = np.where(self.state[:,:,agent] == 1)
            self.state[x,y,agent] = 0 
            self.state[x + dx,y + dy,agent] = 1

    def show(self) -> None:
        """
        Display heatmap of current fire intensities
        """
        IPython.display.clear_output(wait=True)
        cm = mcolors.LinearSegmentedColormap.from_list('', ["gray", "green", "red"], N=100)
        plt.figure(figsize=(3,3))
        plt.imshow(self.state[:,:,INTENSITY], cmap=cm, interpolation='none', vmin=-10, vmax=10)
        plt.colorbar()
        plt.title(f"Fire Spread (t={self.time})")

        for agent in AGENTS:
            x, y = np.where(self.state[:, :, agent] == 1)
            plt.scatter(x, y, c='white')
        plt.show()
        
    def _get_neighbours(self, intensity: np.array) -> tuple:
        """
        Intensity of neighbours: sum of intensity
        Non-zero neighbours: count of active, burning neighbours
        """
        sum_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        binary_intensity = (intensity > 0).astype(int)

        intensity_of_neighbours = convolve2d(intensity, sum_kernel, mode='same', boundary='fill', fillvalue=0)        
        non_zero_neighbours = convolve2d(binary_intensity, sum_kernel, mode='same', boundary='fill', fillvalue=0)
        
        return intensity_of_neighbours, non_zero_neighbours

    def _get_new_ignitions(self, intensity: np.array, intensity_of_neighbours: np.array, non_zero_neighbours: np.array) -> np.array:
        """
        Find empty cells
        Ignite probabilistically based on number of adjacent fires
        """
        ignition_decision = np.random.rand(*intensity.shape) < non_zero_neighbours / 8.0 * IGNITION_RATE
        new_ignitions = (intensity == 0) & ignition_decision
        
        return np.where(new_ignitions, intensity_of_neighbours/non_zero_neighbours*FIRST_JUMP_RATIO, intensity)
    
    def get_done(self):
        """
        False if 1 or more cells are actively burning
        True if all cells are extinguished or alive
        """
        return np.sum(self.state[:, :, INTENSITY] > 0) == 0
    
    def get_reward(self):
        """
        Negative reward for each actively burning cell
        """
        return -np.sum(self.state[:, :, INTENSITY] > 0)
    
    def get_info(self):
        """
        Placeholder for now
        """
        return {}
    
    def get_state(self):
        """
        Return normalized state
        """

def make_board(size=MAP_SIZE, start_intensity=START_INTENSITY, num_points=STARTING_FIRES):
    board = np.zeros((size, size))
    
    # Randomly determine the center and size of the extinguished block
    x, y, width, height = np.random.randint([size // 4, size // 4, size // 4, size // 4],
                                            [3*size // 4, 3*size // 4, size // 2, size // 2])
    
    # Calculate the bounds of the extinguished block, ensuring they are within the board limits
    start_x, end_x = np.clip([x - width // 2, x + (width + 1) // 2], 0, size)
    start_y, end_y = np.clip([y - height // 2, y + (height + 1) // 2], 0, size)
    
    # Set the extinguished block
    board[start_y:end_y, start_x:end_x] = -10

    # Set starting fires without looping, ensuring unique locations
    zero_indices = np.column_stack(np.where(board == 0))
    if len(zero_indices) < num_points:
        # Prevent choosing more points than available zeros
        num_points = len(zero_indices)

    fire_indices = np.random.choice(len(zero_indices), size=num_points, replace=False)
    for index in fire_indices:
        board[tuple(zero_indices[index])] = start_intensity
    
    return board

def dimensions():
    """
    Output observation space as N*N*L for N*N map with L layers
    """
    return (MAP_SIZE, MAP_SIZE, NUM_LAYERS)
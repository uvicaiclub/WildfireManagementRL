import numpy as np
import scipy
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display

MAP_SIZE = 90 # Each unit is 50m x 50m
START_INTENSITY = 3 / 6
STARTING_FIRES = 2

# Agent names: Snap, Crackle, and Pop
# Or Alvin, Simon, and Theodore
INTENSITY, FUEL, MOISTURE, ELEVATION, SELF_EXTINGUISH, WIND_X, WIND_Y, HUMIDITY, NUM_LAYERS = 0, 1, 2, 3, 4, 5, 6, 7, 8

IGNITION_RATE = 0.4 # Increase the time required for the fire to spread
FIRST_JUMP_RATIO = 1.1

AGENT_AREA = 3
ADD_CENTER_MOISTURE = 0.4
ADD_SURROUNDING_MOISTURE = 0.2

FUEL_VARIANCE = 2
MOISTURE_VARIANCE = 2
FUEL_CONSUMPTION_RATE = 0.02

MOISTURE_VISIBILITY = 0.3
MOISTURE_INTENSITY_DAMP = 0
MOISTURE_SPREAD_DAMP = 0.9

# For normalization
MAX_INTENSITY = 6 # Corresponding to 6 ranks of fire
MIN_FUEL = 0.25
MIN_MOISTURE = 0.2
MIN_ELEVATION = 0
MIN_HUMIDITY = 0.25

KERNEL_SIZE = 7
KERNEL_INTENSITY = 0.7
KERNEL_INTENSITY_WEIGHT = 0.5
KERNEL_BASIC_AXIS = 0.2
KERNEL_MINOR_SLOPE = 0.1
KERNEL_MAJOR_SLOPE = 0.3

class FireMap:
    """
    State values:
    0: Intensity of fire [0-6]
    1: Fuel available (starts at 100; burn ~intensity each day)
    2: Moisture
    3: Elevation (not priority)
    4: Cell has self-extinguished
    5,6: Wind (x,y), will be a scalar across entire map
    7: Humidity, will be a scalar across entire map
    """
    def __init__(self, board: np.array) -> None:
        self.state = np.zeros((MAP_SIZE, MAP_SIZE, NUM_LAYERS))
        self.state[:, :, INTENSITY] = board
        self.state[:, :, FUEL] = (1 - MIN_FUEL) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_FUEL
        self.state[:, :, MOISTURE] = (1 - MIN_MOISTURE) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_MOISTURE
        self.state[:, :, ELEVATION] = (1 - MIN_ELEVATION) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_ELEVATION
        self.state[:, :, HUMIDITY] = (1 - MIN_HUMIDITY) * np.random.random() + MIN_HUMIDITY

        wind_x, wind_y = self._init_wind() # Establish unit vector of wind (in any quadrant)
        self.state[:, :, WIND_X] = wind_x
        self.state[:, :, WIND_Y] = wind_y
        self.kernel = FireMap._generate_kernel(wind_x, wind_y)

        self.prev_actions = []
        self.time = 0
        self.game_over = False

    def _init_wind(self) -> tuple[float, float]:
        wind_direction = np.random.random() * 2 * np.pi
        wind_x = np.cos(wind_direction)
        wind_y = np.sin(wind_direction)
        return wind_x, wind_y

    def next(self, actions: list[tuple[float, float]]) -> None:
        intensity = self.state[:, :, INTENSITY]
        fuel = self.state[:, :, FUEL]
        moisture = self.state[:, :, MOISTURE]
        # elevation = self.state[:, :, ELEVATION]

        self._make_actions(actions)
        self.prev_actions = actions
        
        intensity_of_neighbours, non_zero_neighbours = FireMap._get_neighbours(intensity, self.kernel)
        intensity = FireMap._get_new_ignitions(intensity, intensity_of_neighbours, non_zero_neighbours, moisture)
        intensity = np.where(fuel > 0, intensity, -1)

        self.state[:, :, INTENSITY] = np.clip(intensity, -1, 1)
        self.state[:, :, FUEL] -= self.state[:, :, INTENSITY] * FUEL_CONSUMPTION_RATE
        self.time += 1
        
    def _make_actions(self, actions: list[tuple[float, float]]) -> None:
        """
        Add moisture at agent positions
        
        Can only apply moisture to rank 3 or lower.
        If trying in a rank 4 or higher, the simulation ends

        Applying moisture more targeted at center of 3x3 area of effect
        """
        positions = [(AGENT_AREA*x, AGENT_AREA*y) for (x, y) in actions]
        for (x, y) in positions:
            if self.state[x, y, INTENSITY] > 3 / 6:
                self.game_over = True
                break
            else:
                self.state[x:x+AGENT_AREA, y:y+AGENT_AREA, MOISTURE] = np.clip(self.state[x:x+AGENT_AREA, y:y+AGENT_AREA, MOISTURE] + ADD_SURROUNDING_MOISTURE, 0, 1)
                self.state[x+1, y+1, MOISTURE] = np.clip(self.state[x+1, y+1, MOISTURE] + ADD_CENTER_MOISTURE, 0, 1)

    def show(self) -> None:
        """
        Display heatmap of current fire intensities
        Including markers for active agents
        """
        IPython.display.clear_output(wait=True)
        intensity_cm = mcolors.LinearSegmentedColormap.from_list('', ["gray", "green", "red"], N=100)
        moisture_cm = mcolors.LinearSegmentedColormap.from_list('', ["white", "darkblue"], N=100)
        plt.figure(figsize=(3,3))
        plt.imshow(self.state[:,:,INTENSITY], cmap=intensity_cm, interpolation='none', vmin=-1, vmax=1)
        plt.colorbar()
        plt.imshow(self.state[:,:,MOISTURE], cmap=moisture_cm, interpolation='none', alpha=MOISTURE_VISIBILITY, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.title(f"w=({self.state[0, 0, WIND_X]:0.1f}, {self.state[0, 0, WIND_Y]:0.1f}) (t={self.time})")

        wind_x = self.state[0, 0, WIND_X]
        wind_y = self.state[0, 0, WIND_Y]

        # plt.imshow(self.kernel)

        plt.quiver(MAP_SIZE / 2, MAP_SIZE / 2, wind_x, wind_y, scale=3, color='black', width=0.02, headwidth=3, headlength=4)



        for (x, y) in self.prev_actions:
            plt.scatter(AGENT_AREA*x + 1, AGENT_AREA*y + 1, c='white')
        plt.show()
        
    @staticmethod
    def _get_neighbours(intensity: np.array, kernel: np.array) -> tuple[np.array, np.array]:
        """
        Intensity of neighbours: sum of intensity
        Non-zero neighbours: count of active, burning neighbours
        """
        binary_intensity = (intensity > 0).astype(int)

        intensity_of_neighbours = scipy.signal.convolve2d(intensity, kernel, mode='same', boundary='fill', fillvalue=0)        
        non_zero_neighbours = scipy.signal.convolve2d(binary_intensity, kernel, mode='same', boundary='fill', fillvalue=0)
        
        return intensity_of_neighbours, non_zero_neighbours

    @staticmethod
    def _get_new_ignitions(intensity: np.array, intensity_of_neighbours: np.array, non_zero_neighbours: np.array, moisture: np.array) -> np.array:
        """
        Find empty cells
        Ignite probabilistically based on number of adjacent fires
        """
        adjusted_intensity_jump = np.where(non_zero_neighbours > 0, intensity_of_neighbours/non_zero_neighbours *FIRST_JUMP_RATIO*(1 - moisture * MOISTURE_INTENSITY_DAMP), 0)
        # adjusted_intensity_jump = np.rint(adjusted_intensity_jump * 6) / 3

        ignition_decision = np.random.random((MAP_SIZE, MAP_SIZE)) < non_zero_neighbours / 8.0 * IGNITION_RATE * (1 - moisture * MOISTURE_SPREAD_DAMP)
        new_ignitions = (intensity == 0) & ignition_decision
        
        return np.where(new_ignitions, adjusted_intensity_jump, intensity)

    def get_done(self) -> bool:
        """
        Game over if no burning cells
        Or if one agent dies
        """
        return self.game_over or np.sum(self.state[:, :, INTENSITY] > 0) == 0
    
    def get_reward(self) -> int:
        """
        Negative reward for each actively burning cell
        Large negative reward if an agent dies
        """
        return - np.sum(self.state[:, :, INTENSITY] > 0) - 10000 * self.game_over
    
    def get_info(self) -> dict:
        return {}

    @staticmethod
    def _generate_kernel(windx: float, windy: float, wind_speed: float = 1, intensity: float = KERNEL_INTENSITY) -> np.array:
        wind_angle = np.rad2deg(np.arctan2(windy, windx))
        _intensity_bonus = intensity * KERNEL_INTENSITY_WEIGHT
        major_axis = KERNEL_BASIC_AXIS + KERNEL_MAJOR_SLOPE * wind_speed + _intensity_bonus
        minor_axis = KERNEL_BASIC_AXIS + KERNEL_MINOR_SLOPE * wind_speed + _intensity_bonus

        return FireMap._new_kernel(major_axis, minor_axis, angle_deg = wind_angle) * intensity
    
    @staticmethod
    def _create_rotated_distribution(mean, cov_matrix, rot_matrix):    
        # Rotate the covariance matrix
        rotated_cov_matrix = np.dot(rot_matrix,
                                    np.dot(cov_matrix, rot_matrix.T))
        
        # Create a multivariate normal distribution with the rotated covariance matrix
        rotated_distribution = scipy.stats.multivariate_normal(mean=mean,
                                                            cov=rotated_cov_matrix)
        
        return rotated_distribution

    @staticmethod
    def _new_kernel(major_axis: float, minor_axis: float, center: tuple[float, float] = (0, 0), angle_deg: float = 35, kernel_size: int = KERNEL_SIZE) -> np.array:
        if major_axis < minor_axis:
            major_axis, minor_axis = minor_axis, major_axis
        # process angle
        angle_rad = np.deg2rad(angle_deg)
            
        # define the covariance matrix
        cov_matrix = np.array([[major_axis**2, 0], [0, minor_axis**2]])
        
        # find foci
        c = np.sqrt(major_axis**2 - minor_axis**2)
        focus = [c,0]
        
        # define the rotation matrix 
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        
        # rotate the foci and covariance matrix
        rotated_focus = np.dot(rotation_matrix,focus)
        rotated_distribution = FireMap._create_rotated_distribution(mean = rotated_focus,
                                                        cov_matrix = cov_matrix,
                                                        rot_matrix = rotation_matrix)
        
        # sample from the rotated distribution at kernel positions
        x, y = np.meshgrid(np.linspace(-2, 2, kernel_size),
                        np.linspace(-2, 2, kernel_size))
        pos = np.dstack((x, y))
        ellipse = rotated_distribution.pdf(pos)
        
        # Normalize the ellipse
        ellipse /= np.max(ellipse)
            
        return ellipse

def make_board(size: int = MAP_SIZE, start_intensity: int = START_INTENSITY, num_points: int = STARTING_FIRES):
    board = np.zeros((size, size))
    
    # # Randomly determine the center and size of the extinguished block
    # x, y, width, height = np.random.randint([size // 4, size // 4, size // 4, size // 4],
    #                                         [3*size // 4, 3*size // 4, size // 2, size // 2])
    
    # # Calculate the bounds of the extinguished block, ensuring they are within the board limits
    # start_x, end_x = np.clip([x - width // 2, x + (width + 1) // 2], 0, size)
    # start_y, end_y = np.clip([y - height // 2, y + (height + 1) // 2], 0, size)
    
    # # Set the extinguished block
    # board[start_y:end_y, start_x:end_x] = -10

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
import numpy as np
import scipy
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display

MAP_SIZE = 90 # Each unit is 50m x 50m
START_INTENSITY = 1
STARTING_FIRES = 2

# Agent names: Snap, Crackle, and Pop
# Or Alvin, Simon, and Theodore
INTENSITY, FUEL, MOISTURE, ELEVATION, SELF_EXTINGUISH, WIND_X, WIND_Y, HUMIDITY, NUM_LAYERS = 0, 1, 2, 3, 4, 5, 6, 7, 8

IGNITION_RATE = 0.1 # Increase the time required for the fire to spread
FUEL_CONSUMPTION_RATE = 0.1
TEMP_CHANGE_RATE = 0.5
FIRST_JUMP_RATIO = 1.5

AGENT_AREA = 3
ADD_CENTER_MOISTURE = 0.4
ADD_SURROUNDING_MOISTURE = 0.2

FUEL_VARIANCE = 4
MOISTURE_VARIANCE = 2

MOISTURE_VISIBILITY = 0.3
MOISTURE_INTENSITY_DAMP = 0.6
MOISTURE_SPREAD_DAMP = 0.93
MOISTURE_DECAY_RATE = 5e-3
HUMIDITY_SCALAR = 0.1
MOISTURE_INTENSITY_SCALAR = 5e-2
MOISTURE_KERNEL_SCALAR = 0.1
MOISTURE_DRY_RATE = 1e-2

# For normalization
MAX_INTENSITY = 6 # Corresponding to 6 ranks of fire
MIN_FUEL = 0
MIN_MOISTURE = 0.2
MIN_ELEVATION = 0

KERNEL_SIZE = 7
KERNEL_INTENSITY = 0.7
KERNEL_INTENSITY_WEIGHT = 0.5
KERNEL_BASIC_AXIS = 0.2
KERNEL_MINOR_SLOPE = 0.1
KERNEL_MAJOR_SLOPE = 0.3

# clip to max intensity when fuel is below threshold
CLIP_1 = 0.1
CLIP_2 = 0.2
CLIP_3 = 0.3
CLIP_4 = 0.4

class FireMap:
    """
    State values:
    0: Intensity of fire [0-6]
    1: Fuel available (starts at 100%; burn ~intensity each day)
    2: Moisture
    3: Elevation (not a priority)
    4: Cell has self-extinguished
    5,6: Wind (x,y), scalar across entire map
    7: Humidity, scalar across entire map
    """
    def __init__(self, board: np.array) -> None:
        # Initialize state object
        self.state = np.zeros((MAP_SIZE, MAP_SIZE, NUM_LAYERS))
        self.state[:, :, INTENSITY] = board
        self.state[:, :, FUEL] = (1 - MIN_FUEL) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_FUEL
        self.state[:, :, MOISTURE] = (1 - MIN_MOISTURE) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_MOISTURE
        # self.state[:, :, SELF_EXTINGUISH] = ...
        self.state[:, :, ELEVATION] = (1 - MIN_ELEVATION) * np.random.random((MAP_SIZE, MAP_SIZE)) + MIN_ELEVATION
        self.state[:, :, HUMIDITY] = np.random.random()

        # Establish wind unit vector and fixed wind kernel
        wind_x, wind_y = self._init_wind()
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
        # elevation = self.state[:, :, ELEVATION]

        self._make_actions(actions)
        self.prev_actions = actions
        moisture = self.state[:, :, MOISTURE]
        
        intensity_of_neighbours, non_zero_neighbours = FireMap._get_neighbours(intensity, self.kernel)
        intensity = FireMap._get_new_intensity(intensity, moisture, intensity_of_neighbours, non_zero_neighbours)
        self.state[:, :, MOISTURE] = moisture + (self.state[:,:,HUMIDITY] - moisture) * MOISTURE_DECAY_RATE * np.random.random((MAP_SIZE, MAP_SIZE)) - np.clip(intensity + intensity_of_neighbours,0,1) * MOISTURE_DRY_RATE

        conditions = [
            fuel <= 0,
            fuel < CLIP_1,
            fuel < CLIP_2,
            fuel < CLIP_3,
            fuel < CLIP_4
        ]

        choices = [
            -1,  # If fuel <= 0, intensity is set to -1
            np.clip(intensity, -1, 1 / MAX_INTENSITY),  # If fuel < CLIP_1
            np.clip(intensity, -1, 2 / MAX_INTENSITY),  # If fuel < CLIP_2
            np.clip(intensity, -1, 3 / MAX_INTENSITY),  # If fuel < CLIP_3
            np.clip(intensity, -1, 4 / MAX_INTENSITY)   # If fuel < CLIP_4
        ]

        # Default case if none of the conditions are met
        default = np.clip(intensity, -1, 1)

        # snap to nearest multiple of 6
        self.state[:, :, INTENSITY] = np.round(np.select(conditions, choices, default=default) * 6) / 6
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
        colors = ["gray", "gray", "gray", "gray", "green", "yellow", "orange", "lightcoral", "red"]
        nodes = [-1, 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]  # Define nodes for color transitions
        intensity_cm = mcolors.LinearSegmentedColormap.from_list("", colors, N=256)
        moisture_cm = mcolors.LinearSegmentedColormap.from_list('', ["white", "darkblue"], N=100)
        plt.figure(figsize=(3,3))
        plt.imshow(self.state[:,:,INTENSITY], cmap=intensity_cm, interpolation='none', vmin=-1, vmax=1)
        plt.colorbar()
        plt.imshow(self.state[:,:,MOISTURE], cmap=moisture_cm, interpolation='none', alpha=MOISTURE_VISIBILITY, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.title(f"(hum={self.state[0,0,HUMIDITY]:0.1f}) (t={self.time})")

        wind_x = self.state[0, 0, WIND_X]
        wind_y = self.state[0, 0, WIND_Y]

        plt.quiver(MAP_SIZE / 2, MAP_SIZE / 2, wind_x, wind_y, scale=3, color='black', width=0.02, headwidth=3, headlength=4)

        for (x, y) in self.prev_actions:
            plt.scatter(AGENT_AREA*x + 1, AGENT_AREA*y + 1, c='white')
        plt.show()

    @staticmethod
    def _get_neighbours(intensity: np.array, kernel: np.array) -> tuple[np.array, np.array]:
        binary_intensity = (intensity > 0).astype(int)
        intensity_of_neighbours = scipy.signal.convolve2d(intensity, kernel, mode='same', boundary='fill', fillvalue=0)        
        non_zero_neighbours = scipy.signal.convolve2d(binary_intensity, kernel, mode='same', boundary='fill', fillvalue=0)
        return intensity_of_neighbours, non_zero_neighbours

    @staticmethod
    def _get_new_intensity(intensity: np.array, moisture: np.array, intensity_of_neighbours: np.array, non_zero_neighbours: np.array) -> np.array:
        coin_flips = np.random.random((MAP_SIZE, MAP_SIZE))
        old_increase = (intensity > 0) & (coin_flips < TEMP_CHANGE_RATE * (1 + intensity_of_neighbours) * (1 - moisture * MOISTURE_SPREAD_DAMP) / intensity)
        old_decrease = (intensity > 0) & (coin_flips < TEMP_CHANGE_RATE / (1 + intensity_of_neighbours) / (1 - moisture * MOISTURE_SPREAD_DAMP) * intensity)

        # Higher number of neighbours and lower moisture -> higher chance of igniting
        new_ignitions = (intensity == 0) & (coin_flips < non_zero_neighbours * IGNITION_RATE * (1 - moisture * MOISTURE_SPREAD_DAMP))
        new_ignition_intensity = np.where(non_zero_neighbours > 0, np.clip(intensity_of_neighbours / non_zero_neighbours * FIRST_JUMP_RATIO * (1 - moisture * MOISTURE_INTENSITY_DAMP), 1/MAX_INTENSITY, 1), 0)
        
        increased_intensity = np.where(old_increase, intensity + 1/MAX_INTENSITY, intensity)
        decreased_intensity = np.where(old_decrease, np.clip(intensity - 1/MAX_INTENSITY, 0, 1), increased_intensity)

        # Then, apply the new ignition intensity for new ignitions
        final_intensity = np.where(new_ignitions, new_ignition_intensity, decreased_intensity) 
        return final_intensity

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
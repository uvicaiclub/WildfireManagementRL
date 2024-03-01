import numpy as np
import torch as T
import scipy
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display
from odds_and_ends.fireside_bonus import calc_fireside_bonus, calc_fireside_grid

device = T.device("cuda" if T.cuda.is_available() else "cpu")

with open('fire_simulation_settings.json', 'r') as f:
    settings = json.load(f)

# Use settings from the loaded JSON
MAP_SIZE = settings["MAP_SIZE"]
START_INTENSITY = settings["START_INTENSITY"]
STARTING_FIRES = settings["STARTING_FIRES"]
MAP_SIZE = settings["MAP_SIZE"] # Each unit is 50m x 50m
START_INTENSITY = settings["START_INTENSITY"]
STARTING_FIRES = settings["STARTING_FIRES"]

# Agent names: Snap, Crackle, and Pop
# Or Alvin, Simon, and Theodore
INTENSITY = settings["INTENSITY"]
FUEL = settings["FUEL"]
MOISTURE = settings["MOISTURE"]
ELEVATION = settings["ELEVATION"]
SELF_EXTINGUISH = settings["SELF_EXTINGUISH"] 
WIND_X = settings["WIND_X"] 
WIND_Y = settings["WIND_Y"] 
HUMIDITY = settings["HUMIDITY"]
NUM_LAYERS = settings["NUM_LAYERS"]
IGNITION_RATE = settings["IGNITION_RATE"] # Increase the time required for the fire to spread
FUEL_CONSUMPTION_RATE = settings["FUEL_CONSUMPTION_RATE"]
TEMP_CHANGE_RATE = settings["TEMP_CHANGE_RATE"]
FIRST_JUMP_RATIO = settings["FIRST_JUMP_RATIO"]
MOISTURE_DRY_RATE = settings["MOISTURE_DRY_RATE"]
MOISTURE_DECAY_RATE = settings["MOISTURE_DECAY_RATE"]
AGENT_AREA = settings["AGENT_AREA"]
ADD_CENTER_MOISTURE = settings["ADD_CENTER_MOISTURE"]
ADD_SURROUNDING_MOISTURE = settings["ADD_SURROUNDING_MOISTURE"]
FUEL_VARIANCE = settings["FUEL_VARIANCE"]
MOISTURE_VARIANCE = settings["MOISTURE_VARIANCE"]

SHOW_AGENTS = settings["SHOW_AGENTS"]
MOISTURE_VISIBILITY = settings["MOISTURE_VISIBILITY"]
MOISTURE_INTENSITY_DAMP = settings["MOISTURE_INTENSITY_DAMP"]
MOISTURE_SPREAD_DAMP = settings["MOISTURE_SPREAD_DAMP"]
HUMIDITY_SCALAR = settings["HUMIDITY_SCALAR"]
MOISTURE_INTENSITY_SCALAR = settings["MOISTURE_INTENSITY_SCALAR"]
MOISTURE_KERNEL_SCALAR = settings["MOISTURE_KERNEL_SCALAR"]

# For normalization
MAX_INTENSITY = settings["MAX_INTENSITY"] # Corresponding to 6 ranks of fire
MIN_FUEL = settings["MIN_FUEL"]
MIN_MOISTURE = settings["MIN_MOISTURE"]
MIN_ELEVATION = settings["MIN_ELEVATION"]

DRYING_KERNEL_SIZE = settings["DRYING_KERNEL_SIZE"]
KERNEL_SIZE = settings["KERNEL_SIZE"]
KERNEL_INTENSITY = settings["KERNEL_INTENSITY"]
KERNEL_INTENSITY_WEIGHT = settings["KERNEL_INTENSITY_WEIGHT"]
KERNEL_BASIC_AXIS = settings["KERNEL_BASIC_AXIS"]
KERNEL_MINOR_SLOPE = settings["KERNEL_MINOR_SLOPE"]
KERNEL_MAJOR_SLOPE = settings["KERNEL_MAJOR_SLOPE"]

# clip to max intensity when fuel is below threshold
CLIP_1 = settings["CLIP_1"]
CLIP_2 = settings["CLIP_2"]
CLIP_3 = settings["CLIP_3"]
CLIP_4 = settings["CLIP_4"]

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
        self.state[:, :, HUMIDITY] = 0.6

        # Establish wind unit vector and fixed wind kernel
        wind_x, wind_y = self._init_wind()
        self.state[:, :, WIND_X] = wind_x
        self.state[:, :, WIND_Y] = wind_y
        self.kernel = FireMap._generate_kernel(wind_x, wind_y)
        self.dry_kernel = FireMap._generate_kernel(wind_x, wind_y, DRYING_KERNEL_SIZE)

        self.prev_actions = []
        self.time = 0
        self.game_over = False
        self.ring = None

        self.prev_state = self.state

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

        drying_affect = scipy.signal.convolve2d(np.clip(intensity,0,1), self.dry_kernel, mode='same', boundary='fill', fillvalue=0)  
        large_dilation = scipy.ndimage.binary_dilation(abs(self.state[:,:,INTENSITY]), iterations=20, origin=0)
        small_dilation = scipy.ndimage.binary_dilation(abs(self.state[:,:,INTENSITY]), iterations=6)
        dilation_mask = (large_dilation ^ small_dilation) # binary
        self.ring = np.where(dilation_mask, np.clip(drying_affect,0,1000) * (1 - self.state[:,:,MOISTURE]), 0)
        # self.ring = np.clip(drying_affect,0,1000)


        self.state[:, :, MOISTURE] = moisture + (self.state[:,:,HUMIDITY] - moisture) * MOISTURE_DECAY_RATE - np.clip(drying_affect,0,1) * MOISTURE_DRY_RATE

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
        colors = ["gray", "gray", "gray", "gray", "gray", "gray", "green", "yellow", "gold", "orange", "darkorange", "red", "darkred"]
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

        if SHOW_AGENTS:
            for (x, y) in self.prev_actions:
                plt.scatter(AGENT_AREA*y + 1, AGENT_AREA*x + 1, c='white')
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
    
    def get_reward(self, actions) -> float:
        """
        Reward function:
        When we're not at an edge case, we calculate rewards based on how good the action was for the environment.
        We calculate:
            1. How much moisture was added to the environment at all.
            2. How much difference did the action taken have to the active fires
            3. How close the fires are to the edge of the environment.
        """
        if self.game_over:
            return -100
        
        elif self.time > 300:
            return -50
        
        elif np.sum(self.state[:, :, INTENSITY] > 0) == 0:
            return 50
 
        else:
            # Get our moisture map from the previous state
            # moisture_map = np.abs(self.prev_state[:, :, MOISTURE] - 1)
            # moisture_map = np.array(np.where(self.state[:, :, INTENSITY] == -1, 0, moisture_map))

            # moisture_added = 0.0
            # intensity_decreased = 0.0
            # fires = np.array(np.where(self.state[:, :, INTENSITY] > 0))
            # fires_pos = list(zip(fires[0], fires[1]))

            # for action_x, action_y in actions:
            #     for x in [-1, 0, 1]:
            #         for y in [-1, 0, 1]:
            #             try: 
            #                 action_aoe = (action_x + x, action_y + y)
            #                 moisture_added += moisture_map[action_aoe[0], action_aoe[1]]
            #                 if action_aoe in fires_pos:
            #                     #print(intensity_decreased)
            #                     intensity_decreased += 1.5
            #             except IndexError:
            #                 pass

            # if intensity_decreased < 1.0:
            #     intensity_decreased = -5

            # #plt.imshow(moisture_map)
            # #plt.colorbar()
            # #plt.show()

            # # Calculate reward based on distance fires are to the edge
            # fires -= 45
            # max_distance = np.max(np.abs(fires.flatten()))

            # if max_distance >= 45:
            #     self.game_over = True
            #     max_distance *= 10

            firemap_bonus = calc_fireside_bonus(self.prev_state, actions)


            #print(f'Moist: {moisture_levels = }')
            #print(f'BURNT: {burnt_levels = }')
            #print(f'Fire: {fire_levels = }')

            # todo: add component to measure how on fire on average the map is. 
            # not relevant right now because 6 fires are common

            if self.time > 1:
                self.prev_state = self.state

            total = firemap_bonus
            return total
    
    def get_info(self) -> dict:
        return self.ring / np.sum(self.ring)

    @staticmethod
    def _generate_kernel(windx: float, windy: float, kernel_size: float = KERNEL_SIZE, wind_speed: float = 1, intensity: float = KERNEL_INTENSITY) -> np.array:
        wind_angle = np.rad2deg(np.arctan2(windy, windx))
        _intensity_bonus = intensity * KERNEL_INTENSITY_WEIGHT
        major_axis = KERNEL_BASIC_AXIS + KERNEL_MAJOR_SLOPE * wind_speed + _intensity_bonus
        minor_axis = KERNEL_BASIC_AXIS + KERNEL_MINOR_SLOPE * wind_speed + _intensity_bonus

        return FireMap._new_kernel(major_axis, minor_axis, kernel_size, angle_deg = wind_angle) * intensity
    
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
    def _new_kernel(major_axis: float, minor_axis: float, kernel_size: int, center: tuple[float, float] = (0, 0), angle_deg: float = 35) -> np.array:
        if major_axis < minor_axis:
            major_axis, minor_axis = minor_axis, major_axis
        # process angle
        angle_rad = np.deg2rad(np.array(angle_deg))
            
        # define the covariance matrix
        cov_matrix = np.array([[major_axis**2, 0], [0, minor_axis**2]])
        
        # find foci
        c = np.array(np.sqrt(major_axis**2 - minor_axis**2))
        focus = [c,0]
        
        # define the rotation matrix 
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        
        # rotate the foci and covariance matrix
        rotated_focus = np.array(np.dot(rotation_matrix,focus))
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

    fire_indices = [(np.random.randint(35, 65), np.random.randint(35, 65)) for _ in range(num_points)]

    for x, y in fire_indices:
        board[x, y] = start_intensity
    
    return board

def dimensions():
    """
    Output observation space as N*N*L for N*N map with L layers
    """
    return (MAP_SIZE, MAP_SIZE, NUM_LAYERS)
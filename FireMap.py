import numpy as np
import math
from scipy.signal import convolve2d
from kernels import IntensityKernel

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display

MAP_SIZE = 90 # Each unit is 50m x 50m
START_INTENSITY = 3 / 6
STARTING_FIRES = 2

# Agent names: Snap, Crackle, and Pop
# Or Alvin, Simon, and Theodore
INTENSITY, FUEL, MOISTURE, ELEVATION, SELF_EXTINGUISH, WIND_X, WIND_Y, HUMIDITY, NUM_LAYERS = 0, 1, 2, 3, 4, 5, 6, 7, 8

IGNITION_RATE = 1 # Increase the time required for the fire to spread
FIRST_JUMP_RATIO = 1.2 

AGENT_AREA = 3
ADD_CENTER_MOISTURE = 0.4
ADD_SURROUNDING_MOISTURE = 0.2

FUEL_VARIANCE = 2
MOISTURE_VARIANCE = 2
FUEL_CONSUMPTION_RATE = 0.05

MOISTURE_VISIBILITY = 0.3
MOISTURE_INTENSITY_DAMP = 0
MOISTURE_SPREAD_DAMP = 0.9

# For normalization
MAX_INTENSITY = 6 # Corresponding to 6 ranks of fire
MIN_FUEL = 0.25
MIN_MOISTURE = 0.2
MIN_ELEVATION = 0
MIN_HUMIDITY = 0.25


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

        # Set previous state to calculate reward
        self.prev_state = self.state

        wind_x, wind_y = self._init_wind() # Establish unit vector of wind (in any quadrant)
        self.state[:, :, WIND_X] = wind_x
        self.state[:, :, WIND_Y] = wind_y
        rounded_wind = FireMap._round_to_8_directions(wind_x, wind_y)
        ik = IntensityKernel()
        self.kernel = ik.get_kernel(rounded_wind)

        self.prev_actions = []
        self.time = 0
        self.game_over = False

    def _init_wind(self) -> tuple[float, float]:
        wind_direction = np.random.random() * math.pi/2
        wind_x = math.cos(wind_direction) * (-1 if np.random.random() > 0.5 else 1)
        wind_y = math.sin(wind_direction) * (-1 if np.random.random() > 0.5 else 1)
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
        #print(np.where(self.state[:, :, INTENSITY] == -1))
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

    def show(self, iter) -> None:
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

        rounded_wind_x, rounded_wind_y = FireMap._round_to_8_directions(wind_x, wind_y)
        # You can adjust the starting point of the vector. Here, it's set to the center of the plot.
        origin_x = MAP_SIZE / 2
        origin_y = MAP_SIZE / 2
        # Scale the wind vector for visibility if needed, especially if wind_x and wind_y are very small.
        scale_factor = 3  # Adjust as needed based on your wind speed values
        plt.quiver(origin_x, origin_y, rounded_wind_x, rounded_wind_y, scale=scale_factor, color='black', width=0.02, headwidth=3, headlength=4)
        # plt.quiver(origin_x, origin_y, wind_x, wind_y, scale=scale_factor, color='black', width=0.02, headwidth=3, headlength=4)



        for (x, y) in self.prev_actions:
            plt.scatter(AGENT_AREA*x + 1, AGENT_AREA*y + 1, c='white')

        name_of_file = f'firemap_iter_{iter}'
        plt.savefig(fname=f"odds_and_ends/runs/{name_of_file}")
        #plt.show()
        plt.close()
        
    @staticmethod
    def _get_neighbours(intensity: np.array, kernel: np.array) -> tuple[np.array, np.array]:
        """
        Intensity of neighbours: sum of intensity
        Non-zero neighbours: count of active, burning neighbours
        """
        binary_intensity = (intensity > 0).astype(int)

        intensity_of_neighbours = convolve2d(intensity, kernel, mode='same', boundary='fill', fillvalue=0)        
        non_zero_neighbours = convolve2d(binary_intensity, kernel, mode='same', boundary='fill', fillvalue=0)
        
        return intensity_of_neighbours, non_zero_neighbours

    @staticmethod
    def _get_new_ignitions(intensity: np.array, intensity_of_neighbours: np.array, non_zero_neighbours: np.array, moisture: np.array) -> np.array:
        """
        Find empty cells
        Ignite probabilistically based on number of adjacent fires
        """
 
        #plt.imshow(non_zero_neighbours)
        #plt.show()
        
        # adjusted_intensity_jump = np.where(non_zero_neighbours > 0, intensity_of_neighbours/non_zero_neighbours *FIRST_JUMP_RATIO*(1 - moisture* MOISTURE_INTENSITY_DAMP), 0)
        adjusted_intensity_jump = np.where(non_zero_neighbours > 0, (intensity_of_neighbours)*FIRST_JUMP_RATIO, 0)
        # adjusted_intensity_jump = np.rint(adjusted_intensity_jump * 6) / 3

        ignition_decision = np.random.random((MAP_SIZE, MAP_SIZE)) < non_zero_neighbours / 8.0 * IGNITION_RATE * (1 - moisture * MOISTURE_SPREAD_DAMP)
        new_ignitions = (intensity == 0) & ignition_decision
        
        return np.where(new_ignitions, adjusted_intensity_jump, intensity)
    
    @staticmethod
    def _round_to_8_directions(x, y):
        angle = np.arctan2(y, x)
        direction_idx = int(np.ceil(4/np.pi*(angle + 2*np.pi- np.pi/8)))
        direction_map = [
            (1, 0),   # East
            (1, 1),   # Northeast
            (0, 1),   # North
            (-1, 1),  # Northwest
            (-1, 0),  # West
            (-1, -1), # Southwest
            (0, -1),  # South
            (1, -1)   # Southeast
        ]
        return direction_map[direction_idx % 8]

    def get_done(self) -> bool:
        """
        Game over if no burning cells
        Or if one agent dies
        """
        return self.game_over or np.sum(self.state[:, :, INTENSITY] > 0) == 0 or self.time > 300
    
    def get_reward(self, actions) -> int:
        """
        Reward function:
        When we're not at an edge case, we calculate rewards based on how good the action was for the environment.
        We calculate:
            1. How much moisture was added to the environment at all.
            2. How much difference did the action taken have to the active fires
            3. How close the fires are to the edge of the environment.
        """

        # TODO: Isolate rewards on edges, edge penalty. 
        if self.game_over:
            # Danger Penalty: Running into an active fire = bad.
            return -100
        
        elif self.time > 300:
            return -50
        
        elif np.sum(self.state[:, :, INTENSITY] > 0) == 0:
            return 50

        else:
            # Get our moisture map from the previous state
            # ---- Moisture Opprotunity Bonus ----
            moisture_map = np.abs(self.prev_state[:, :, MOISTURE] - 1)
            moisture_map = np.array(np.where(self.state[:, :, INTENSITY] == -1, 0, moisture_map))

            moisture_added = 0.0
            intensity_decreased = 0.0
            fires = np.array(np.where(self.state[:, :, INTENSITY] > 0))
            fires_pos = list(zip(fires[0], fires[1]))

            for action_x, action_y in actions:
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        try: 
                            action_aoe = (action_x + x, action_y + y)
                            moisture_added += moisture_map[action_aoe[0], action_aoe[1]]
                            if action_aoe in fires_pos:
                                #print(intensity_decreased)
                                intensity_decreased += 1.5
                        except IndexError:
                            pass

            if intensity_decreased < 1.0:
                intensity_decreased = -5

            #plt.imshow(moisture_map)
            #plt.colorbar()
            #plt.show()

            # Calculate reward based on distance fires are to the edge
                
            # ---- Distance to Edge Penalty ----
            fires -= 45
            max_distance = np.max(np.abs(fires.flatten()))

            if max_distance >= 45:
                self.game_over = True
                max_distance *= 10


            #print(f'Moist: {moisture_levels = }')
            #print(f'BURNT: {burnt_levels = }')
            #print(f'Fire: {fire_levels = }')

            # todo: add component to measure how on fire on average the map is. 
            # not relevant right now because 6 fires are common

            total = -1*max_distance

            #print(moisture_added*1.5, ' ---- ', intensity_decreased, ' ---- ', -1*max_distance)

            # reset prev.state
            if self.time > 1:
                self.prev_state = self.state
            else:
                pass
            #print(f'{total = }')

            return total
    
    def get_info(self) -> dict:
        return {}

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

    # Centering starting fires
    # TODO: refactor
    fire_indices = np.random.choice(30, size=6, replace=False) + 30 
    
    for x, y in zip(fire_indices[:3], fire_indices[3:]):
        board[x, y] = start_intensity
    
    return board

def dimensions():
    """
    Output observation space as N*N*L for N*N map with L layers
    """
    return (MAP_SIZE, MAP_SIZE, NUM_LAYERS)
import numpy as np
from scipy.signal import convolve2d
#from kernels import IntensityKernel

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy

INTENSITY = 0

pretend_map = np.zeros((90,90,8))
pretend_map[15,10,0] = 1
pretend_map[25,44,0] = 1
pretend_map[15,34,0] = 1
    
def np_maxpool2D(mat: np.array, pool=3) -> np.array:
    ''' preserves the channels layer '''
    M, N,channels = mat.shape
    K = L = pool
    MK = M // K
    NL = N // L
    return mat[:MK*K, :NL*L].reshape(MK, K, NL, L, channels).max(axis=(1, 3))

def random_action():
    temp = np.random.randint(30,size=6)
    return list(zip(temp[:3], temp[3:]))

def calc_fireside_grid(obs):
    # TODO: stitch this into the existing work
    # "obs" is the full state (intensity is extracted in this function)
    # actions is a list of tuples
    
    ''' --- euc distance --- '''
    # calculate euclidean distance (first panel)
    x, y = np.meshgrid(np.linspace(-45, 45, 90),
                       np.linspace(-45, 45, 90))
    pos = np.dstack((x, y))
    # euc distances for all points in mesh
    euc_distance_clipped = np.clip(np.linalg.norm(pos,axis=2)-10,0,120)
    
    ''' --- dilation mask --- '''
    #if obs.shape[0] == 90:
    #    obs = np_maxpool2D(obs,pool=3)

    # why are there negative intensities?
    fires_clipped = np.clip(obs[:,:,INTENSITY],a_min=0,a_max=1) > 0
    dilation_mask = scipy.ndimage.binary_dilation(fires_clipped, iterations=9)

    ''' --- remove "on-fire" --- '''
    fire_removed = dilation_mask.astype(int) - (obs[:,:,INTENSITY] > 0)*2
    
    ''' --- intersection --- '''
    fireside_available = euc_distance_clipped * fire_removed
    return fireside_available


def calc_fireside_bonus(obs, actions) -> float:
    # TODO: stitch this into the existing work
    # "obs" is the full state (intensity is extracted in this function)
    # actions is a list of tuples
    
    ''' --- euc distance --- '''
    # calculate euclidean distance (first panel)
    x, y = np.meshgrid(np.linspace(-45, 45, 90),
                       np.linspace(-45, 45, 90))
    pos = np.dstack((x, y))
    # euc distances for all points in mesh
    euc_distance_clipped = np.clip(np.linalg.norm(pos,axis=2)-10,0,120)
    
    ''' --- dilation mask --- '''
    #if obs.shape[0] == 90:
    #    obs = np_maxpool2D(obs,pool=3)

    # why are there negative intensities?
    fires_clipped = np.clip(obs[:,:,INTENSITY],a_min=0,a_max=1) > 0
    dilation_mask = scipy.ndimage.binary_dilation(fires_clipped, iterations=9)

    ''' --- remove "on-fire" --- '''
    fire_removed = dilation_mask.astype(int) - (obs[:,:,INTENSITY] > 0)*2
    
    ''' --- intersection --- '''
    fireside_available = euc_distance_clipped * fire_removed
    
    ''' --- bonus based on action --- '''
    bonus = 0
    for _x, _y in actions:
        bonus += fireside_available[_x,_y] / 25.0

    if bonus == 0.0:
        bonus = -1

    #print(bonus)
    return bonus
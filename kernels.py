import numpy as np

class IntensityKernel:
    '''
    To minimize the computation needed, we can create a kernel that highlights how
    likely a certain cell (always centered at the center) will catch on fire. This can 
    simultaneously be used to calculate the probability a certain cell will increase 
    in intensity.

    We create 8 possible kernels for the eight possible wind directions. Depending
    on what direction the wind takes (and possibly scaling for intensity), we 
    retrieve one of the eight possible kernel operations to layer over our entire map.
    '''
    def __init__(self):
        ''' 
        We identify, 
        1, 0 = <-
        -1, 0 = ->
        so on and so forth.

        The values in the kernels themselves are normalized between 0 and 1.
        '''
        self.kernels = {(1, 0): [[0.3, 0.5, 0.8],
                                 [0.3, 1.0, 1.0],
                                 [0.3, 0.5, 0.8]], 
                        (-1, 0): [[0.8, 0.5, 0.3],
                                 [1.0, 1.0, 0.3],
                                 [0.8, 0.5, 0.3]], 
                        (0, -1): [[0.8, 1.0, 0.8],
                                 [0.5, 1.0, 0.5],
                                 [0.3, 0.3, 0.3]], 
                        (0, 1): [[0.3, 0.3, 0.3],
                                 [0.5, 1.0, 0.5],
                                 [0.8, 1.0, 0.8]],
                        (1, -1): [[0.5, 0.8, 1.0],
                                 [0.3, 1.0, 0.8],
                                 [0.3, 0.3, 0.5]],
                        (1, 1): [[0.3, 0.3, 0.5],
                                 [0.3, 1.0, 0.8],
                                 [0.5, 0.8, 1.0]],
                        (-1, -1): [[1.0, 0.8, 0.5],
                                 [0.8, 1.0, 0.3],
                                 [0.5, 0.3, 0.3]],
                        (-1, 1): [[0.5, 0.3, 0.3],
                                 [0.8, 1.0, 0.3],
                                 [1.0, 0.8, 0.5]]}
        
        

    def get_kernel(self, wind_dir: tuple) -> np.array:
        return np.array(self.kernels[wind_dir])
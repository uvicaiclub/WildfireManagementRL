import numpy as np
from torch import nn
import torch

def process_env_for_agent(obs, fireside) -> np.array:
  '''
  take a NxNxL environment state, reduce it down for RL agent's input
  NOTE: hardcoded for 8 layers and 90x90.
  '''
  # scale layers
  pass # nathan has kept everything squeezed between 0 and 1

  print(obs.shape)

  # reduce dimensions
  # (sorry. this swaps the axis of the state to put layers first, pools, adds batch representation, and fixes a transposition that got in there somehow)
  pooling_layer = nn.MaxPool2d(kernel_size=3, stride=3)
  obs_red = np.swapaxes(obs,2,0).reshape(1,8,90,90)
  obs_red = obs_red[:, :, :3]
  obs_red = np.concatenate(obs, fireside)

  # discretize observations
  # don't let agents see perfect resolution of "moisture"
  pass

  return obs_red


def process_RL_outputs(raw_actions) -> list[tuple]:
  
  ''' takes the tensor from the NN, converts it to a list of three tuples in environment resolution
  NOTE: baord size is hard-coded
  '''
  _n_agents = len(raw_actions)//2

  # rescale to environment resolution
  #raw_actions *= 3


  # clip to min and max (0 and 30)
  raw_actions = np.clip(raw_actions,a_min=0,a_max=89)
  raw_actions = raw_actions.astype(int)

  # combine actions into 
  # (all x positions, then all y positions)

  return list(zip(raw_actions[:_n_agents]//1, raw_actions[_n_agents:]//1))


# example usage
# the environment is a (90,90,8)
full_scale = np.random.rand(90,90,8)
reduced = process_env_for_agent(obs = full_scale) # reduced is (1,8,30,30)

# NN(reduced) -> raw_actions
raw_actions = np.random.rand(6)*40 - 5
process_RL_outputs(raw_actions)

import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time
import tqdm
import matplotlib.pyplot as plt

from PPO_Agent_Misc.PPOContinuous import Agent as ContPPO
from Simulation import FireMapEnv
from odds_and_ends.data_processing import process_RL_outputs, process_env_for_agent
from odds_and_ends.fireside_bonus import calc_fireside_bonus, calc_fireside_grid

# HYPERPARAMETERS
EPOCHS = 25
NUM_MINIBATCHES = 32
MINIBATCH = 16
EPISODES = 500
TS_PER_ITER = 2000

# Continuous Parameters
action_min = T.tensor((0.0, -1.0))
action_max = T.tensor((1.0, 1.0))

PPO_Agent = ContPPO(n_actions=6, c1=0.5, c2=0.5, input_dims=8464, action_min=action_min, action_max=action_max, 
                    gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=MINIBATCH, 
                    buffer_size=MINIBATCH*NUM_MINIBATCHES, n_epochs=EPISODES, LR=1e-3, annealing=False)

#PPO_Agent.load('PPO_Agent_Misc/Agent_weights_actor', 'PPO_Agent_Misc/Agent_weights_critic')

env = FireMapEnv()
env.reset()

actions = np.reshape(np.random.randint(0, 45/3, 6), (3,2))
obs_1, rewards, dones, info = env.step(actions)

full_episode_loss = []
avg_policy_loss = []
avg_crit_loss = []
episode_max_ratio = []
ep_mean_rewards = []

fireside = calc_fireside_grid(obs_1)
#obs = process_env_for_agent(obs, fireside)
obs = T.unsqueeze(T.unsqueeze(T.tensor(fireside, dtype=T.float32), dim=0), dim=0).to(PPO_Agent.device)
print(obs.shape)
env.reset()
for e in tqdm.tqdm(range(EPISODES)):
    # Step one, get some sort of training running. 
    for _ in range(TS_PER_ITER):
        # Set prev observation
        
        prev_obs = obs.clone()
        
        # Get actions from the agent
        actions, logprob, mean, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)

        # Process raw actions into the environment 
        processed_actions = process_RL_outputs(actions.numpy())
        
        #print(processed_actions)
        obs_1, rewards, dones, _ = env.step(processed_actions)
        fireside = calc_fireside_grid(obs_1)
        #obs = process_env_for_agent(obs, fireside)
        obs = T.unsqueeze(T.unsqueeze(T.tensor(fireside, dtype=T.float32), dim=0), dim=0).to(PPO_Agent.device)

        rewards = T.tensor(rewards, dtype=T.float32)
        if dones == True:
            env.reset()

        # Get value of obs and store in agent
        next_vf = PPO_Agent.critic.forward(obs)
        next_vf = next_vf.detach()
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, actions, logprob, advantage, prev_vf, rewards, dones)

        # Learning loop
        if len(PPO_Agent.memory.states) >= NUM_MINIBATCHES*MINIBATCH:
            e_policy_loss, e_crit_loss, loss = PPO_Agent.learn()

        #env.render()

        PPO_Agent.c2 *= 0.95
        PPO_Agent.actor.var *= 0.99

    print("Episode , ", e)

PPO_Agent.save('PPO_Agent_Misc/Agent_weights')

env.reset()
for _ in range(30):
    for _ in range(10):
        prev_obs = obs.clone()
    
        # Get actions from the agent
        actions, logprob, mean, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        
        print(actions)
        # Process raw actions into the environment 
        processed_actions = process_RL_outputs(actions.numpy())
        
        print(processed_actions)
        obs, rewards, dones, _ = env.step(processed_actions)
        fireside = calc_fireside_grid(obs)
        #obs = process_env_for_agent(obs, fireside)
        obs = T.unsqueeze(T.unsqueeze(T.tensor(fireside, dtype=T.float32), dim=0), dim=0).to(PPO_Agent.device)

        if dones == True:
            env.reset()

        rewards = T.tensor(rewards, dtype=T.float32)
        #print(f"{rewards = }")

        env.render()

env.close()



import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO


env = gym.make("LunarLander-v2", render_mode='human')


observation, info = env.reset(seed=42)

PPO_Agent = PPO(4, 0.4, 0.4, 8)

action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)
for _ in range(20):
    for e in range(64*2):
        prev_obs = T.tensor(obs).clone().detach()
        action, log_prob, entropy, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        obs, rewards, dones, info, _ = env.step(np.array(action))
        obs = T.tensor(obs)

        next_vf = PPO_Agent.critic.forward(T.tensor(obs))
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, action, log_prob, advantage, prev_vf, rewards, dones)

        if len(PPO_Agent.memory.states) >= 64:
            avg_reward = PPO_Agent.learn()
            print(avg_reward)

        if e == 0:
            print("cur state", np.array(obs).shape)
        env.render()

    env.reset(seed=42)
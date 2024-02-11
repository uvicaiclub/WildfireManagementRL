import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO


env = gym.make("LunarLander-v2", render_mode='human')


observation, info = env.reset(seed=42)

PPO_Agent = PPO(4, 0.4, 0.4, 8)

full_episode_loss = []

action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)


obs = T.tensor(obs)
for episode in range(10):
    episode_loss = 0.0
    for e in range(64*10):
        prev_obs = obs.clone().detach()
        action, log_prob, entropy, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        obs, rewards, dones, info, _ = env.step(np.array(action))
        obs = T.tensor(obs)

        next_vf = PPO_Agent.critic.forward(T.tensor(obs))
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, action, log_prob, advantage, prev_vf, rewards, dones)

        if dones == True:
            env.reset(seed=42)

        if len(PPO_Agent.memory.states) >= 64:
            avg_reward = PPO_Agent.learn(obs)
            episode_loss += np.array(avg_reward.detach())

        if e == 0:
            print("cur state", np.array(obs).shape)
        env.render()

    print("Episode: ", episode)
    print()
    full_episode_loss.append(episode_loss)

    env.reset(seed=42)


env.close()
print(full_episode_loss)
plt.plot(full_episode_loss)
plt.show()
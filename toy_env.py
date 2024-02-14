import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO


env = gym.make("LunarLander-v2", render_mode='human')


observation, info = env.reset(seed=42)

EPOCHS = 10

PPO_Agent = PPO(n_actions=4, c1=1.0, c2=0.01, input_dims=8, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, buffer_size=64*5, n_epochs=10, LR=1e-4)

action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)

full_episode_loss = []
avg_policy_loss = []
avg_crit_loss = []


obs = T.tensor(obs)
for episode in range(10):
    episode_loss = 0.0
    episode_policy_loss = 0.0
    episode_crit_loss = 0.0
    for e in range(64*EPOCHS):
        prev_obs = obs.clone().detach()
        action, log_prob, entropy, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        obs, rewards, dones, info, _ = env.step(np.array(action))
        obs = T.tensor(obs)

        next_vf = PPO_Agent.critic.forward(T.tensor(obs))
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, action, log_prob, advantage, prev_vf, rewards, dones)

        if dones == True:
            env.reset(seed=42)

        if len(PPO_Agent.memory.states) >= 64*5:
            for _ in range(5):
                e_policy_loss, e_crit_loss, avg_reward = PPO_Agent.learn(obs)
                episode_loss += np.array(avg_reward)
                episode_crit_loss += np.array(e_crit_loss)
                episode_policy_loss += np.array(e_policy_loss)

        if e == 0:
            print("cur state", np.array(obs).shape)
        env.render()

    print("Episode: ", episode)
    print()
    full_episode_loss.append(episode_loss/EPOCHS)
    avg_crit_loss.append(episode_crit_loss/EPOCHS)
    avg_policy_loss.append(episode_policy_loss/EPOCHS)

    env.reset(seed=42)


env.close()
print(full_episode_loss)
plt.plot(full_episode_loss, label="Episode Loss")
plt.plot(avg_crit_loss, label="Critic Loss")
plt.plot(avg_policy_loss, label="Policy Loss")
plt.legend()
plt.title("Training Losses over Time")
plt.xlabel("Episode")
plt.ylabel("Loss Magnitude")
plt.show()
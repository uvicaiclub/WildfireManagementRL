import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO
import seaborn as sns


env = gym.make("LunarLander-v2")


episode_seed = np.random.randint(0, 100)
observation, info = env.reset(seed=episode_seed)

# HYPERPARAMETERS
EPOCHS = 25
NUM_MINIBATCHES = 32
MINIBATCH = 16
ITERATIONS = 400
TS_PER_ITER = 2000

PPO_Agent = PPO(n_actions=4, c1=1.0, c2=0.01, input_dims=8, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=MINIBATCH, buffer_size=MINIBATCH*NUM_MINIBATCHES, n_epochs=ITERATIONS, LR=1e-4)

action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)

full_episode_loss = []
avg_policy_loss = []
avg_crit_loss = []
episode_max_ratio = []
ep_mean_rewards = []

obs = T.tensor(obs)
for episode in range(ITERATIONS):
    episode_loss = 0.0
    episode_policy_loss = 0.0
    episode_crit_loss = 0.0
    ep_max_ratio = 0.0
    ep_tot_rewards = 0.0
    for e in range(TS_PER_ITER):
        prev_obs = obs.clone().detach()
        action, log_prob, entropy, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        obs, rewards, dones, info, _ = env.step(np.array(action))
        obs = T.tensor(obs)

        ep_tot_rewards += rewards

        next_vf = PPO_Agent.critic.forward(obs)
        next_vf = next_vf.detach()
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, action, log_prob, advantage, prev_vf, rewards, dones)

        if dones == True:
            env.reset(seed=episode_seed)

        if len(PPO_Agent.memory.states) >= 64*5:
            for _ in range(5):
                e_policy_loss, e_crit_loss, loss = PPO_Agent.learn(obs)
                episode_loss += np.array(loss)
                episode_crit_loss += np.array(e_crit_loss)
                episode_policy_loss += np.array(e_policy_loss)
                

        # Render the env
        #if episode > 350:
            #env.render()

    print("Episode: ", episode)
    print()
    full_episode_loss.append(episode_loss/TS_PER_ITER)
    avg_crit_loss.append(episode_crit_loss/TS_PER_ITER)
    avg_policy_loss.append(episode_policy_loss/TS_PER_ITER)
    episode_max_ratio.append(ep_max_ratio)
    ep_mean_rewards.append(ep_tot_rewards/TS_PER_ITER)
    #print(episode_max_ratio)

    episode_seed = np.random.randint(0, 100)
    env.reset(seed=episode_seed)

env.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].plot(ep_mean_rewards, label="Episode Mean Rewards")
ax[0].legend()
ax[0].grid()
ax[0].set_title("Rewards Per Episode")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Rewards")

ax[1].plot(full_episode_loss, label="Episode Loss")
ax[1].plot(avg_crit_loss, label="Critic Loss")
ax[1].plot(avg_policy_loss, label="Policy Loss")

ax[1].legend()
ax[1].grid()
ax[1].set_title("Training Losses over Time")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Loss Magnitude")

plt.show()
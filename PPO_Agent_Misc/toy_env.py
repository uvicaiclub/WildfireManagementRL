import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO
from PPOContinuous import Agent as ContPPO
import seaborn as sns
import tqdm


env = gym.make("CarRacing-v2")

episode_seed = np.random.randint(0, 100)
observation, info = env.reset(seed=episode_seed)

# HYPERPARAMETERS
EPOCHS = 25
NUM_MINIBATCHES = 32
MINIBATCH = 16
EPISODES = 400
TS_PER_ITER = 2000

# Continuous Parameters
action_min = T.tensor((0.0, -1.0))
action_max = T.tensor((1.0, 1.0))


PPO_Agent = ContPPO(n_actions=3, c1=0.5, c2=0.1, input_dims=9216, action_min=action_min, action_max=action_max, 
                    gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=MINIBATCH, 
                    buffer_size=MINIBATCH*NUM_MINIBATCHES, n_epochs=EPISODES, LR=1e-3, annealing=False)

action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)
print(obs.shape)

full_episode_loss = []
avg_policy_loss = []
avg_crit_loss = []
episode_max_ratio = []
ep_mean_rewards = []

obs = T.tensor(np.expand_dims(np.swapaxes(obs, 2, 0), axis=0)).float().to(PPO_Agent.device)
print(obs.shape)

for episode in tqdm.tqdm(range(EPISODES)):
    episode_loss = 0.0
    episode_policy_loss = 0.0
    episode_crit_loss = 0.0
    ep_max_ratio = 0.0
    ep_tot_rewards = 0.0
    for e in range(TS_PER_ITER):
        prev_obs = obs.clone().detach()
        action, logprob, mean, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
        obs, rewards, dones, info, _ = env.step(action.numpy())

        obs = T.tensor(np.expand_dims(np.swapaxes(obs, 2, 0), axis=0)).float().to(PPO_Agent.device)

        next_vf = PPO_Agent.critic.forward(obs)
        advantage = PPO_Agent.get_gae(rewards, prev_vf, next_vf)
        PPO_Agent.memory.store_memory(prev_obs, action, logprob, advantage, prev_vf, T.tensor(rewards).to(PPO_Agent.device), dones)
        #print(action)

        if dones == True:
            env.reset(seed=episode_seed)

        if len(PPO_Agent.memory.states) >= NUM_MINIBATCHES*MINIBATCH:
            e_policy_loss, e_crit_loss, loss = PPO_Agent.learn()
            episode_loss += np.array(loss)
            episode_crit_loss += np.array(e_crit_loss)
            episode_policy_loss += np.array(e_policy_loss)

            PPO_Agent.c2 *= 0.95
            PPO_Agent.actor.var *= 0.999
                
        ep_tot_rewards += rewards

        # Render the env

    #print("Episode: ", episode)
    #print()
    full_episode_loss.append(episode_loss/TS_PER_ITER)
    avg_crit_loss.append(episode_crit_loss/TS_PER_ITER)
    avg_policy_loss.append(episode_policy_loss/TS_PER_ITER)
    episode_max_ratio.append(ep_max_ratio)
    ep_mean_rewards.append(ep_tot_rewards/TS_PER_ITER)
    #print(episode_max_ratio)

    episode_seed = np.random.randint(0, 100)
    env.reset(seed=episode_seed)

env.close()


# Render games at the end
env = gym.make("CarRacing-v2", render_mode="human")

env.reset()
action = env.action_space.sample()
obs, rewards, dones, info, _ = env.step(action)

obs = T.tensor(np.expand_dims(np.swapaxes(obs, 2, 0), axis=0)).float().to(PPO_Agent.device)

for e in range(TS_PER_ITER):
    prev_obs = obs.clone().detach()
    action, logprob, mean, prev_vf = PPO_Agent.get_action_and_vf(prev_obs)
    print(action)
    obs, rewards, dones, info, _ = env.step(np.array(action))
    obs = T.tensor(np.expand_dims(np.swapaxes(obs, 2, 0), axis=0)).float().to(PPO_Agent.device)

    ep_tot_rewards += rewards

    if dones == True:
        env.reset()
            
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
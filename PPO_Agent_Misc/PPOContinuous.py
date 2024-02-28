import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time

device = T.device("cuda" if T.cuda.is_available() else "cpu")
T.autograd.set_detect_anomaly(True)

class relu30(nn.Module):
    def __init__(self):
        super(relu30, self).__init__()

    def forward(self, x):
        return T.min(T.max(T.tensor(0), x), T.tensor(30))
        

class PPOMemory:
    def __init__(self, batch_size):
        # we keep memory in mind with lists
        self.states = []
        self.actions = []
        self.logprobs = []
        self.adv = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, adv, vals, reward, done):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(probs.detach())
        self.adv.append(adv.detach())
        self.vals.append(vals.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.logprobs = []
        self.vals = []
        self.adv = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def get_memory_batch(self):
        ''' Returns a memory batch of size batch_size. '''

        # retrieves batch_size memories
        states_T = T.stack(self.states[:self.batch_size]).to(device)
        states_T = T.squeeze(states_T, dim=1)
        act_logprob_tens = T.stack(self.logprobs[:self.batch_size]).to(device)
        adv_tensor = T.tensor(self.adv[:self.batch_size]).to(device)
        vals_tens = T.tensor(self.vals[:self.batch_size], dtype=T.float64).to(device)
        act_tens = T.stack(self.actions[:self.batch_size]).to(device)
        rew_tens = T.stack(self.rewards[:self.batch_size]).to(device)
        done_tens = T.tensor(self.dones[:self.batch_size]).to(device)

        # removes the first self.batch_size memories.
        del self.states[:self.batch_size]
        del self.logprobs[:self.batch_size]
        del self.adv[:self.batch_size]
        del self.vals[:self.batch_size]
        del self.actions[:self.batch_size]
        del self.rewards[:self.batch_size]
        del self.dones[:self.batch_size]

        return states_T, act_logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens, done_tens
    
# ---- Actor and Critic Models ---
    
class ActorModel(nn.Module):
    '''
        A continuous actionspace forces us to create some sort of distribution
        on the range of values on our actionspace. We force a normal distribution.
        Functionally what this means is our mean and var values are parameters 
        that change with training, so we have to create some sort of basis for this.
        
        Variational Autoencoders split a layer at the same depth to account for 
        this exact same thing. The code will look similar.

    '''
    def __init__(self, input_shape, n_actions, c2,
                 min_tens = T.tensor((-1, 1)), max_tens = T.tensor((-1, 1))):
        super(ActorModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.flat = nn.Flatten()

        # base model
        self.fc1 = nn.Linear(in_features=input_shape, out_features=642).to(device)
        self.fc2 = nn.Linear(in_features=642, out_features=128).to(device)

        # distributions
        self.mean = nn.Linear(in_features=128, out_features=n_actions).to(device)

        # Constant variance
        self.c2 = c2
        self.var = T.diag(T.ones(n_actions)).to(device)*0.5

        # misc
        self.min_tens = min_tens
        self.max_tens = max_tens
        self.relu30 = relu30()

    def forward(self, x):
        ''' We create a class that computes the distribution of our actions. '''

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.flat(x)
        # base computation
        x = F.tanh(self.fc1(x).to(device))
        x = F.tanh(self.fc2(x)).to(device)

        # we note the mean value goes through a tanh as it corresponds with the 
        # lunar lander task at hand. Specifically, we have a output which ranges from (-1,1)

        # for our environment, depending on how we represent the raw actions, 
        # we can have positive and negative values, if we center in the middle of the board
        # for example.
        mean = F.tanh(self.mean(x)).to(device)
        #activation = self.relu30(mean)

        return T.squeeze(mean)
    
    def get_action_logprob(self, x):
        mean = self.forward(x)

        calc_mean = mean.to(device)
        #print(f'{calc_mean = }')

        mean = T.clamp(mean, 0, 29)

        # get action from distribution distribution
        dist = T.distributions.MultivariateNormal(calc_mean, self.var)

        action = dist.sample()
        logprob = dist.log_prob(action)
        

        return action.to(device), logprob, mean.to(device)
    
    def calc_action_logprob_entropy(self, x, action):
        mean = self.forward(x)
        calc_mean = mean.to(device)

        dist = T.distributions.MultivariateNormal(calc_mean, self.var)

        logprob = dist.log_prob(action)
        ent = dist.entropy()

        return logprob, ent
    
    def calc_c2(self):
        new_c2 = self.c2*0.95
        self.c2 = max(new_c2, 0.5)
    
class CriticModel(nn.Module):
    def __init__(self, input_shape):
        super(CriticModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(input_shape, 642).to(device)
        self.fc2 = nn.Linear(642, 128).to(device)
        self.output = nn.Linear(128, 1).to(device)

    def forward(self, x):
        '''
        Overwrites the basic call function
        '''
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.flat(x)
        x = F.tanh(self.fc1(x)).to(device)
        x = F.tanh(self.fc2(x)).to(device)
        x = self.output(x).to(device)

        return T.squeeze(x)

class Agent(nn.Module):
        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
    
    def __init__(self, n_actions, c1, c2, input_dims, action_min, action_max, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, buffer_size=64*10, n_epochs=10, LR=1e-3,
                 annealing=True):
        
        super(Agent, self).__init__()

        #           --- Hyperparams ---
        self.gamma = gamma
        self.policy_clip = T.tensor(policy_clip, dtype=T.float32)
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_actions = n_actions

        #           --- Actor Critic ---
        self.actor = ActorModel(input_dims, n_actions, 2).float().to(device)
        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), LR)

        self.critic = CriticModel(input_dims).float().to(device)
        self.optimizer_critic = T.optim.Adam(self.critic.parameters(), LR)

        #           --- Memory ---
        self.memory = PPOMemory(batch_size)

        #           --- Misc ---
        self.criterion = nn.MSELoss()
        self.annealing = annealing
        if annealing == True:
            self.anneal_lr_actor = T.optim.lr_scheduler.StepLR(self.optimizer_actor, self.buffer_size, gamma=0.95)
            self.anneal_lr_critic = T.optim.lr_scheduler.StepLR(self.optimizer_critic, self.buffer_size, gamma=0.95)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.training_steps = 0
        self.action_min = action_min
        self.action_max = action_max

        self.prev_val_loss = 1
    
    def get_gae(self, reward, vf_t, vf_t1):
        ''' As seen here: https://arxiv.org/pdf/1506.02438.pdf
            An estimation for the advantage function. 
            GAE = r_t - gamma*lambda*vf_(t+1) + vf(t)
        '''
        return reward - self.gamma*self.gae_lambda*vf_t1 + vf_t
    
    def get_action_and_vf(self, x):
        ''' get action and associated vf '''
        action, logprob, mean = self.actor.get_action_logprob(x)

        return action, logprob, mean, self.critic.forward(x).to(device)
    
    def discount_path(self, path, h):
        """
        Given a "path" of items x_1, x_2, ... x_n, return the discounted
        path, i.e. 
        X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
        X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
        etc.
        Can do (more efficiently?) w SciPy. Python here for readability
        Inputs:
        - path, list/tensor of floats
        - h, discount rate
        Outputs:
        - Discounted path, as above
        """
        curr = 0
        rets = []
        for i in range(len(path)):
            curr = curr*h + path[-1-i]
            rets.append(curr)
        rets =  T.stack(list(reversed(rets)), 0)
        return rets
    
    def advantage_and_return(self, rewards, values, not_dones):
        """
        Calculate GAE advantage, discounted returns, and 
        true reward (average reward per trajectory)

        GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)

        using formula from John Schulman's code:
        V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
        """
        
        V_s_tp1 = T.cat([values[:,1:], values[:, -1:]], 1) * not_dones
        deltas = rewards + self.gamma * V_s_tp1 - values

        # now we need to discount each path by gamma * lam
        advantages = T.zeros_like(rewards)
        returns = T.zeros_like(rewards)
        indices = self.get_path_indices(not_dones)
        for agent, start, end in indices:
            advantages[agent, start:end] = self.discount_path( \
                    deltas[agent, start:end], self.gae_lambda*self.gamma)
            returns[agent, start:end] = self.discount_path( \
                    rewards[agent, start:end], self.gamma)

        return advantages.clone().detach(), returns.clone().detach()

    def calculate_boundary_penalty(self, action_position: T.tensor) -> T.tensor:
        left = np.clip(action_position, a_min=-np.inf,a_max=0)
        right = np.clip(action_position-30,a_min=0,a_max=np.inf)
        return T.tensor(np.sum(np.max(np.vstack([abs(left),abs(right)]),axis=0))).to(device)

    def learn(self):
        '''
        This function iterates over our entire buffer size and trains over minibatches.
        '''

        # We set the variables nessessary to scale our rewards and advantages 
        rew_mean = T.tensor(self.memory.rewards).mean()
        rew_std = T.tensor(self.memory.rewards).std()

        adv_mean = T.tensor(self.memory.adv).mean()
        adv_std = T.tensor(self.memory.adv).std()

        #self.memory.rewards = (self.memory.rewards - rew_mean) / rew_std
        total_pol_loss = 0.0
        total_critic_loss = 0.0
        total_loss = 0.0
        
        for minibatch in range(self.buffer_size//self.batch_size):

            # retrieve memories from last batch
            state_tens, logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens, done_tens = self.memory.get_memory_batch()


            #           --- Actor and Entropy Loss ---


            # get logprob of action and entropy
            new_logprobs, entropy = self.actor.calc_action_logprob_entropy(state_tens, act_tens) 

            entropy = T.mean(entropy)     

            # Get probability raio
            prob_ratios = T.exp(new_logprobs - logprob_tens).to(device)

            # policy loss

            clipped_ratios = T.clamp(prob_ratios, 1-self.policy_clip, 1+self.policy_clip).to(device)
            policy_loss = T.min(prob_ratios, clipped_ratios).to(device)


            #           --- Critic Loss ---


            approx_val = T.flatten(self.critic.forward(state_tens)).to(device)
            #approx_val_clip = T.clamp(approx_val, self.prev_val_loss-self.policy_clip, self.prev_val_loss+self.policy_clip).to(device)

            
            #           --- Advantage and Returns  ---


            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rew_tens), reversed(done_tens)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            # Normalizing the rewards
            rewards = T.tensor(rewards, dtype=T.float32).to(device)
            rewards = (rewards - rew_mean) / (rew_std + 1e-7)

            # Normalize advantages
            adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-10)


            #           --- Total Loss ---

            # Apply Advantages to Policy Loss
            policy_loss = adv_tensor*policy_loss
            policy_loss = T.mean(policy_loss).to(device)

            # Apply returns 

            crit_loss = self.criterion(approx_val, rewards)
            #crit_loss_clipped = self.criterion(approx_val_clip, rewards)
                        
            #crit_loss = T.max(crit_loss, crit_loss_clipped).to(device)
            crit_loss = self.c1*crit_loss.float().to(device)

            loss = -policy_loss + crit_loss #- self.c2*entropy


            #           --- Backpropogate ---


            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            loss.backward()

            self.optimizer_actor.step()
            self.optimizer_critic.step()

            if self.annealing == True:
                self.anneal_lr_actor.step()
                self.anneal_lr_critic.step()

            # Add to total loss
            total_pol_loss += policy_loss.detach().numpy()
            total_critic_loss += crit_loss.detach().numpy()
            total_loss += loss.detach().numpy()
        
        self.actor.calc_c2()


        return total_pol_loss, total_critic_loss, total_loss

        


##A
    
    
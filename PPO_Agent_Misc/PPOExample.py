'''
Actor Critic Methods will often have performance tanks after a certain amount
of time due to being sensitive to perturbations. 
This was the inspiration behind the PPO algorithm. Effectively the process
of making the TRPO algorithm more efficient and less prone to mass fluctuations.

It does this by using what the paper calls 'clipped probability ratios'
which is effectively comparing policies between timesteps to eachother 
with a set lower bound. Basing the update of the policy between some 
ratio of a new policy to the old. The term probability comes due to having
0-1 as bounds.

PPO also keeps 'memories' maybe similar to that of DQN. Multiple updates 
to the network happen per data sample, which are carried out through
minibatch stochastic gradient ascent. 

Implementation notes: Memory
We note that learning in this case is carried out through batches. 
We keep a track of, say, 50 state transitions, then train on a batch 
of 5-10-15 of them. The size of the batch is arbitrary for implementation 
but there likely exists a best batch size. It seems to be the case that 
the batches are carried out from iterative state transfers only. 

Implementation notes: Critic
Two distinct networks instead of shared inputs. 
Actor decides to do based on the current state, and the critic evaluates states.

Critic Loss:
Return = advantage + critic value (from memory).
then the L_critic = MSE(return - critic vlaue (from network))

Networks outputs probabilities for an action distribution, therefore exploration is
handled by definition. 

Overview:
Class for replay buffer, which can be implemented quite well with lists. 
Class for actor network and critic network
Class for the agent, tying everything together
Main loop to train and evaluate

'''

import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time

device = T.device("cuda" if T.cuda.is_available() else "cpu")

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
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(probs)
        self.adv.append(adv)
        self.vals.append(vals)
        self.rewards.append(reward)
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

        # retrieves the 64 states
        states_T = T.stack(self.states[:self.batch_size]).to(device)
        act_logprob_tens = T.tensor(self.logprobs[:self.batch_size]).to(device)
        adv_tensor = T.tensor(self.adv[:self.batch_size]).to(device)
        vals_tens = T.tensor(self.vals[:self.batch_size], dtype=T.float64).to(device)
        act_tens = T.stack(self.actions[:self.batch_size]).to(device)
        rew_tens = T.tensor(self.rewards[:self.batch_size]).to(device)

        # removes the first self.batch_size states
        del self.states[:self.batch_size]
        del self.logprobs[:self.batch_size]
        del self.adv[:self.batch_size]
        del self.vals[:self.batch_size]
        del self.actions[:self.batch_size]
        del self.rewards[:self.batch_size]

        return states_T, act_logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens


class Agent(nn.Module):
        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
    
    def __init__(self, n_actions, c1, c2, input_dims, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, buffer_size=64*10, n_epochs=10, LR=1e-3,
                 annealing=True):
        
        super(Agent, self).__init__()

        #           --- Hyperparams ---
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_actions = n_actions

        #           --- Actor Critic ---
        self.actor = self._create_model(input_dims, n_actions).float().to(device)
        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), LR)

        self.critic = self._create_model(input_dims, 1).float().to(device)
        self.optimizer_critic = T.optim.Adam(self.critic.parameters(), LR)

        #           --- Memory ---
        self.memory = PPOMemory(batch_size)

        #           --- Misc ---
        self.criterion = nn.MSELoss()
        self.annealing = annealing
        if annealing == True:
            self.anneal_lr_actor = T.optim.lr_scheduler.StepLR(self.optimizer_actor, buffer_size*5, gamma=0.3)
            self.anneal_lr_critic = T.optim.lr_scheduler.StepLR(self.optimizer_critic, buffer_size*5, gamma=0.3)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.training_steps = 0
    

    def _create_model(self, input_dims, output_dims):
        ''' private function meant to create the same model with varying input/output dims. '''
        model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dims)
        )
        return model

    def get_vf(self, x):
        ''' retrieve the value function for that state as determined by critic. '''
        return self.critic.forward(x)
    
    def get_gae(self, reward, vf_t, vf_t1):
        ''' As seen here: https://arxiv.org/pdf/1506.02438.pdf
            An estimation for the advantage function. 
            GAE = r_t - gamma*lambda*vf_(t+1) + vf(t)
        '''
        return reward - self.gamma*self.gae_lambda*vf_t1 + vf_t
    
    def get_action_and_vf(self, x):
        ''' get distribution over actions and associated vf '''
        logits = self.actor(x.to(device))
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic.forward(x)

    def learn(self, new_state):
        # retrieve memories from last batch
        state_tens, act_logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens = self.memory.get_memory_batch()

        # Forget gradients from last learning step
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        #           --- Actor and Entropy Loss ---

        # Scale our advantage functions for better convergence
        #print(adv_tensor)
        #adv_tensor = (adv_tensor - adv_tensor.mean()) / adv_tensor.std()
        #print(adv_tensor)

        # Send our state through our actors
        new_probs = Categorical(logits=self.actor(state_tens))

        # get prob of our action
        prob_of_action = new_probs.log_prob(act_tens)

        # Entropy Loss
        entropy_loss = T.mean(new_probs.entropy()).to(device)

        # Get probability raio
        prob_ratios = T.exp(prob_of_action - act_logprob_tens).to(device)

        #maximum_ratio = T.max(prob_ratios).detach().numpy()

        # Clip Max Tensor
        clip_max = T.tensor(1+self.policy_clip, dtype=T.float32).expand(self.batch_size, 1).to(device)

        # Clip Min Tensor
        clip_min =  T.tensor(1-self.policy_clip, dtype=T.float32).expand(self.batch_size, 1).to(device)

        #print("clamped Policies: ", T.clamp(prob_ratios, clip_min, clip_max))
        #print("non clamped policies: ", prob_ratios)

        # policy loss
        policy_loss = T.min((prob_ratios*adv_tensor), T.clamp(prob_ratios, clip_min, clip_max)*adv_tensor).to(device)
        policy_loss = T.mean(policy_loss).to(device)

        #           --- Critic Loss ---

        # sum over the rewards to get returns
        bootstrap = self.critic(new_state.to(device)).detach()
        
        returns = T.cat([rew_tens, bootstrap]).to(device)
        returns = T.flip(returns, dims=(0,)).to(device)

        returns = T.cumsum(returns, dim=0).to(device)
        returns = T.flip(returns, dims=(0,)).to(device)

        returns = (returns) / (returns.std())
        returns = returns.float()

        approx_val = T.flatten(self.critic.forward(state_tens)).to(device)
        
        crit_loss = self.criterion(approx_val, returns[:self.batch_size])

        # Implementation detail in 'Implementation Matters in Deep Policy Gradients'
        crit_loss_clip = T.clamp(crit_loss, 1-self.policy_clip, 1+self.policy_clip).to(device)
        crit_loss = T.min(crit_loss, crit_loss_clip).to(device)
        crit_loss = crit_loss.float().to(device)

        #           --- Total Loss ---

        # Implementation detail in 'Implementation Matters in Deep Policy Gradients'
        # No entropy loss
        loss = -policy_loss + self.c1*crit_loss - self.c2*entropy_loss

        #print('policy_loss: ', policy_loss)
        #print('crit loss: ', crit_loss)
        #print('entropy loss: ', entropy_loss)
        #print(loss)

        # backward pass
        loss.backward()

        self.optimizer_actor.step()
        self.optimizer_critic.step()

        if self.annealing == True:
            self.anneal_lr_actor.step()
            self.anneal_lr_critic.step()
            #print("### Learning Rate : ", self.anneal_lr_actor.get_last_lr() , " ###")
            #self.training_steps += 1

        # Exponential Decay on C2
        self.c2 *= 0.999

        return policy_loss.detach(), crit_loss.detach(), loss.detach()

        


##A
    
    
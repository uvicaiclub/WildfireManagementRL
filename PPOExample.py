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

class Agent(nn.Module):
    def __init__(self, n_actions, c1, c2, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size

        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
        
        '''
        Our actor loss works by comparing our old distribution by our new distribution, then 
        multiplying by our advantage function. For this to work, we need somewhere to store
        the distribution 'pi_old' 
        '''
        self.actor = self._create_model(input_dims, n_actions)

        self.critic = self._create_model(input_dims, 1)
        self.memory = PPOMemory(batch_size)

    def _create_model(self, input_dims, output_dims):
        ''' private function meant to create the same model with varying input/output dims. '''
        return nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims)
        )

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def get_vf(self, x):
        ''' retrieve the value function for that state as determined by critic. '''
        return self.critic.forward(x)
    
    def get_action_and_vf(self, x):
        ''' get distribution over actions and associated vf '''
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic.forward(x)

    def learn(self):
        # retrieve memories from last batch
        state_tens = T.tensor(self.memory.states)
        act_logprob_tens = T.tensor(self.memory.logprobs)
        adv_tensor = T.tensor(self.memory.adv)
        vals_tens = T.tensor(self.memory.vals)
        act_tens = self.memory.actions
        rew_tens = self.memory.rewards
        done_tens = T.tensor(self.memory.dones)

        # clear our memory for the next batch
        self.memory.clear_memory()

        #           --- Actor and Entropy Loss ---

        # Send our state through our actors
        new_probs = Categorical(logits=self.actor(state_tens))

        # get prob of our action
        prob_of_action = new_probs.log_prob(act_tens)

        # Entropy Loss
        entropy_loss = -T.mean(new_probs.entropy())

        # Get probability raio
        prob_ratios = T.exp(prob_of_action - act_logprob_tens)

        # Clip Max Tensor
        clip_max = T.tensor(1+self.policy_clip, dtype=T.float32).expand(self.batch_size, 1)

        # Clip Min Tensor
        clip_min =  T.tensor(1-self.policy_clip, dtype=T.float32).expand(self.batch_size, 1)   

        # policy loss
        policy_loss = min(T.mean(prob_ratios*adv_tensor), T.mean(T.clamp(prob_ratios, clip_min, clip_max)*adv_tensor))

        #           --- Critic Loss ---

        # MSE over the advantages
        crit_loss = nn.functional.mse_loss(adv_tensor)

        #           --- Total Loss ---

        loss = policy_loss + self.c_1*crit_loss + self.c_2*entropy_loss

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        

        pass


##A
    
    
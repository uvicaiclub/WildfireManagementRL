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
        states_T = T.stack(self.states[:64])
        act_logprob_tens = T.tensor(self.logprobs[:64])
        adv_tensor = T.tensor(self.adv[:64])
        vals_tens = T.tensor(self.vals[:64], dtype=T.float64)
        act_tens = T.tensor(self.actions[:64])
        rew_tens = T.tensor(self.rewards[:64])

        # removes the first 64 states
        del self.states[:64]
        del self.logprobs[:64]
        del self.adv[:64]
        del self.vals[:64]
        del self.actions[:64]
        del self.rewards[:64]

        return states_T, act_logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens


class Agent(nn.Module):
        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
    
    def __init__(self, n_actions, c1, c2, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, buffer_size=64*10, n_epochs=10, LR=1e-3):
        
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
        self.actor = self._create_model(input_dims, n_actions)
        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), LR)

        self.critic = self._create_model(input_dims, 1)
        self.optimizer_critic = T.optim.Adam(self.critic.parameters(), LR)

        #           --- Memory ---
        self.memory = PPOMemory(batch_size)

        #           --- Misc ---
        self.criterion = nn.MSELoss()
        self.ts = 0
    

    def _create_model(self, input_dims, output_dims):
        ''' private function meant to create the same model with varying input/output dims. '''
        model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
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
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic.forward(x)

    def learn(self, new_state):
        # retrieve memories from last batch
        state_tens, act_logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens = self.memory.get_memory_batch()

        # Forget gradients from last learning step
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        #print(state_tens, '\n')
        #print(act_logprob_tens, '\n')
        #print(adv_tensor, '\n')
        #print(vals_tens, '\n')
        #print(act_tens, '\n')
        #print(rew_tens, '\n')
        #time.sleep(20)

        #           --- Actor and Entropy Loss ---

        # Send our state through our actors
        new_probs = Categorical(logits=self.actor(state_tens))

        # get prob of our action
        prob_of_action = new_probs.log_prob(act_tens)

        # Entropy Loss
        entropy_loss = T.mean(new_probs.entropy())

        # Get probability raio
        prob_ratios = T.exp(prob_of_action - act_logprob_tens)

        # Clip Max Tensor
        clip_max = T.tensor(1+self.policy_clip, dtype=T.float32).expand(self.batch_size, 1)

        # Clip Min Tensor
        clip_min =  T.tensor(1-self.policy_clip, dtype=T.float32).expand(self.batch_size, 1)  

        print("clamped Policies: ", T.clamp(prob_ratios, clip_min, clip_max))
        print("non clamped policies: ", prob_ratios)

        # policy loss
        policy_loss = T.min((prob_ratios*adv_tensor), T.clamp(prob_ratios, clip_min, clip_max)*adv_tensor)
        policy_loss = T.mean(policy_loss)

        #           --- Critic Loss ---

        # sum over the rewards to get returns
        bootstrap = self.critic(new_state).detach()
        
        returns = T.cat([rew_tens, bootstrap])
        returns = T.flip(returns, dims=(0,))

        returns = T.cumsum(returns, dim=0)
        returns = T.flip(returns, dims=(0,))

        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        crit_loss = self.criterion(vals_tens, returns[:64])

        #           --- Total Loss ---

        loss = -policy_loss + self.c1*crit_loss - self.c2*entropy_loss

        #print('policy_loss: ', policy_loss)
        #print('crit loss: ', crit_loss)
        #print('entropy loss: ', entropy_loss)

        # backward pass
        loss.backward()

        self.optimizer_actor.step()
        self.optimizer_critic.step()

        # Exponential Decay on C2
        self.c2 *= 0.99

        # Making sure the memory works as it is supposed to
        print(len(self.memory.states))

        return policy_loss.detach(), crit_loss.detach(), loss.detach()

        


##A
    
    
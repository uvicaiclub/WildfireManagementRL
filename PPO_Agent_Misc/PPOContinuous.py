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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time

device = T.device("cuda" if T.cuda.is_available() else "cpu")
T.autograd.set_detect_anomaly(True)

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
        act_logprob_tens = T.stack(self.logprobs[:self.batch_size]).to(device)
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
    def __init__(self, input_shape, n_actions, 
                 min_tens = T.tensor((-1, 1)), max_tens = T.tensor((-1, 1))):
        super(ActorModel, self).__init__()

        # base model
        self.fc1 = nn.Linear(in_features=input_shape, out_features=64).to(device)
        self.fc2 = nn.Linear(in_features=64, out_features=64).to(device)

        # distributions
        self.mean = nn.Linear(in_features=64, out_features=n_actions).to(device)

        # Constant variance
        self.var = T.ones(self.n_actions).to(device) #T.diag(self.n_actions).unsqueeze(dim=0)
        print(self.var)

        # misc
        self.min_tens = min_tens
        self.max_tens = max_tens

    def forward(self, x):
        ''' We create a class that computes the distribution of our actions. '''

        # base computation
        x = F.tanh(self.fc1(x)).to(device)
        x = F.tanh(self.fc2(x)).to(device)

        # split
        # we note the mean value goes through a tanh as it corresponds with the 
        # lunar lander task at hand. Specifically, we have a output which ranges from (-1,1)
        mean = F.tanh(self.mean(x)).to(device)

        return mean
    
    def get_action_logprob(self, x):
        mean = self.forward(x)

        calc_mean = mean.data.to(device)

        # get action from distribution distribution
        action = T.normal(calc_mean, self.var)

        if action[0] < 0:
            action[0] = 0

        logprob = self.calc_logprob(calc_mean, self.var, action)

        return action.to(device), logprob, mean.to(device)
    
    def calc_action_logprob_entropy(self, x, action):
        mean = self.forward(x)
        calc_mean = mean.detach().to(device)

        return self.calc_logprob(calc_mean, self.var, action)
    
    def calc_logprob(self, mean, var, actions):
        a = T.square(actions - mean) / 2*self.var
        b = T.log(T.sqrt(2 * np.pi * self.var))
        return a + b
    
    def calc_entropy(self, var):
        return T.sqrt(2*np.pi*np.e*var)
    
class CriticModel(nn.Module):
    '''

    '''
    def __init__(self, input_shape):
        super(CriticModel, self).__init__()

        self.fc1 = nn.Linear(input_shape, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.output = nn.Linear(64, 1).to(device)

    def forward(self, x):
        '''
        Overwrites the basic call function
        '''
        x = F.tanh(self.fc1(x)).to(device)
        x = F.tanh(self.fc2(x)).to(device)
        x = F.tanh(self.output(x)).to(device)

        return x



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
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_actions = n_actions

        #           --- Actor Critic ---
        self.actor = ActorModel(input_dims, n_actions).float().to(device)
        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), LR)

        self.critic = CriticModel(input_dims).float().to(device)
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

    def learn(self, new_state):
        # Forget gradients from last learning step
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        # retrieve memories from last batch
        state_tens, logprob_tens, adv_tensor, vals_tens, act_tens, rew_tens = self.memory.get_memory_batch()

        #           --- Actor and Entropy Loss ---

        # get logprob of action and entropy
        new_logprobs = self.actor.calc_action_logprob_entropy(state_tens, act_tens)        

        # Get probability raio
        prob_ratios = T.exp(new_logprobs - logprob_tens).to(device)

        #maximum_ratio = T.max(prob_ratios).detach().numpy()

        # Clip Max Tensor
        clip_max = T.tensor(1+self.policy_clip, dtype=T.float32).expand(self.batch_size, 2).to(device)

        # Clip Min Tensor
        clip_min =  T.tensor(1-self.policy_clip, dtype=T.float32).expand(self.batch_size, 2).to(device)

        #print("clamped Policies: ", T.clamp(prob_ratios, clip_min, clip_max))
        #print("non clamped policies: ", prob_ratios)

        adv_tensor = T.stack((adv_tensor, adv_tensor))
        adv_tensor = T.transpose(adv_tensor, 0, 1)

        # policy loss
        policy_loss = T.min((prob_ratios), T.clamp(prob_ratios, clip_min, clip_max)).to(device)
        policy_loss = T.mean(policy_loss, axis=1).to(device)
        policy_loss = T.mean(policy_loss, axis=0).to(device)

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
        #crit_loss_clip = T.clamp(crit_loss, self.prev_val_loss-self.policy_clip, self.prev_val_loss+self.policy_clip).to(device)
        #crit_loss = T.min(crit_loss, 10000000000).to(device)
        crit_loss = crit_loss.float().to(device)

        #self.prev_val_loss = np.array(crit_loss)
        #           --- Total Loss ---

        # Implementation detail in 'Implementation Matters in Deep Policy Gradients'
        # No entropy loss
        loss = -policy_loss + self.c1*crit_loss # - self.c2*entropy_loss

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


        return policy_loss.detach(), crit_loss.detach(), loss.detach()

        


##A
    
    
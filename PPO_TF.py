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
import tensorflow as tf

class PPOMemory:
    def __init__(self, batch_size):
        # we keep memory in mind with lists
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), \
               np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class Agent(tf.Module):
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
        self.actor = self._create_model(input_dims, n_actions)
        self.critic = self._create_model(input_dims, 1)
        self.memory = PPOMemory(batch_size)

    def _create_model(self, input_dims, output_dims):
        ''' private function meant to create the same model with varying input/output dims. '''
        xin = tf.keras.layers.Input(input_dims)
        x = tf.keras.layers.Dense(64, activation='relu')(xin)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(output_dims, activation='relu')(x)
        return tf.keras.models.Model(xin, x)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def get_vf(self, x):
        ''' retrieve the value function for that state as determined by critic. '''
        return self.critic.forward(x)
    
    def get_action_and_vf(self, x):
        ''' get distribution over actions and associated vf '''
        logits = self.actor(x)
        probs = tf.random.categorical(logits=logits)
        action = probs.num_sample(1)
        return action, probs.log_prob(action), self.critic(x)
    
    def get_expected_return(self,
        rewards: tf.Tensor) -> tf.Tensor:
        """Compute expected returns per timestep. Used to evaluate critic network."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        return returns




    


    
    
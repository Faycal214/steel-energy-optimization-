import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from memory import PPOMemory

class ActorNetwork(nn.Module) :
    def __init__(self, n_actions, input_dims, alpha, 
                 fc1_dims = 256, fc2_dims = 256) :
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim = -1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr= alpha)
    
    def forward(self, state) :
        state = torch.tensor(state, dtype = torch.float32)
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class CriticNetwork(nn.Module) :
    def __init__(self, input_dims, alpha, 
                 fc1_dims = 256, fc2_dims = 256) :
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    
    def forward(self, state) :
        state = torch.tensor(state, dtype = torch.float32)
        value = self.critic(state)
        return value
    
class PPOAgent :
    def __init__(self, input_dims, n_actions, gamma = 0.99, alpha = 3e-4, gae_lambda = 0.95,
                 policy_clip = 0.2, batch_size = 64, N = 2048, n_epochs = 10) :
        
        self.gamma = gamma
        self.policy_clip = policy_clip 
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.input_dims = input_dims
        
        self.actor = ActorNetwork(n_actions, self.input_dims, alpha)
        self.critic = CriticNetwork(self.input_dims, alpha)
        self.memory = PPOMemory(batch_size)
    
    def remember(self, state, action, probs, vals, reward, done) :
        self.memory.store_memory(state, action, reward, probs, vals, done)
    
    def choose_action(self, observation) :
        state = torch.tensor(observation, dtype = torch.float)
        
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, value
    
    def learn(self) :
        for _ in range(self.n_epochs) :
            state_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, done_arr, batches = \
                    self.memory.generate_batchs()
            
            values = vals_arr
            advantages = np.zeros(len(reward_arr), dtype = np.float32)
            
            for t in range(len(reward_arr) - 1) :
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr )- 1) :
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1])*\
                        (1 - int(done_arr[k]) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantages[t] = a_t
            advantages = torch.tensor(advantages)
            
            values = torch.tensor(values)
            for batch in batches :
                states = torch.tensor(state_arr[batch], dtype = torch.float)
                old_probs = torch.tensor(old_probs_arr[batch])
                actions = torch.tensor(action_arr[batch])
                
                dist = self.actor.forward(states)
                critic_value = self.critic.forward(states)
                critic_value = torch.squeeze(critic_value)
                
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantages[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()
        
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class CriticNetwork(nn.Module) :
    def __init__(self, beta, input_dims, n_actions, fc1_dims = 512, fc2_dims = 256) :
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
                            
        # self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)  # Ensure correct input size
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr= beta)
        
    def forward(self, state, action) :
        action_value = self.fc1(T.cat([action, state], dim= 1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        
        return q
    
class ValueNetwork(nn.Module) :
    def __init__(self, beta, input_dims, fc1_dims = 256, fc2_dims = 256) :
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
    
    def forward(self, state) :
        value = self.fc1(state)
        value = F.relu(value)
        value = self.fc2(value)
        value = F.relu(value)
        v = self.v(value)
        
        return v

class ActorNetwork(nn.Module) :
    def __init__(self, alpha, input_dims, max_action, fc1_dims = 256, fc2_dims = 256, n_actions = 2) :
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.reparam_noise = 1e-6
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
    def forward(self, state) :
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min= self.reparam_noise, max= 1)
                
        return mu, sigma
    
    def sample_normal(self, state, reparameterize = True) :
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        
        if reparameterize :
            actions = probabilities.rsample()
        else :
            actions = probabilities.sample()
        
        action = T.tanh(actions) * T.tensor(self.max_action)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2)+ self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim = True)
        
        return actions, log_probs
import numpy as np
import torch
from memory import Memory
from networks import DeepQNetwork

class dqn :
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, 
                 batch_size, max_memory_size, combined = False, eps_ends = 0.05, epsilon_decay = 5e-5):
        
        self.state_dim = state_dim
        self.action_dim = [i for i in range(action_dim)]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.eps_ends = eps_ends
        self.memory = Memory(state_dim, max_memory_size, batch_size, combined)
        
        # every C time steps (which is replace target) will update the parameters of the target network 
        # starting with the counter 0 
        self.iter_cntr = 0
        self.replace_target = 100
        
        self.Q_eval = DeepQNetwork(state_dim, action_dim, learning_rate)
        self.Q_target = DeepQNetwork(state_dim, action_dim, learning_rate)
    
    def choose_action(self, observation) :
        if np.random.random() > self.epsilon : 
            state = torch.tensor(observation)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else :
            action = np.random.choice(self.action_dim)
        return action
    
    def learn(self) :
        if not self.memory.is_sufficient() :
            return 
        
        self.Q_eval.optimizer.zero_grad()
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        states, actions, rewards, next_states, dones = self.memory.sample_memory()
        states = torch.tensor(states[batch_index])
        rewards = torch.tensor(rewards[batch_index])
        next_states = torch.tensor(next_states[batch_index])
        dones = torch.tensor(dones[batch_index])
        q_eval = self.Q_eval.forward(states)[batch_index, actions]
        q_next = self.Q_target.forward(next_states)
        q_next[dones] = 0.0
        q_target = rewards + (self.gamma * torch.max(q_next, dim= 1)[0])
        
        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_ends else self.eps_ends
        
        if self.iter_cntr % self.replace_target == 0 :
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
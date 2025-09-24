import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module) :
    def __init__(self, input_dim, output_dim) :
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.hc1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.hc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)
    
    def forward(self, state) :
        state = torch.tensor(state)
        x = self.hc1(self.fc1(state))
        x = self.hc2(self.fc2(x))
        action_logits = self.fc3(x)
        return torch.softmax(action_logits, dim= -1)
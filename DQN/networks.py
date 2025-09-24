import torch 
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module) :
    def __init__(self, input_dim, out_dim, learning_rate) :
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(32, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr= learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state) :
        state = torch.tensor(state)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
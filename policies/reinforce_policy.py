import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReinforcePolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=128):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, state):
        state = torch.FloatTensor(state)
        return self.policy_network(state)
    
    def select_action(self, state):
        probs = self.forward(state)
        action = torch.multinomial(probs, 1)
        self.saved_log_probs.append(torch.log(probs[action]))
        return action.item()
    
    def update(self):
        returns = self.compute_returns(gamma=0.99)
        policy_loss = []
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.saved_log_probs = []
        self.rewards = []
        
    def compute_returns(self, gamma):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
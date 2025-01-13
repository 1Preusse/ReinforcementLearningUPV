import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

class SGDPolicy(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Outputs for radius mean and angle mean
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.saved_actions = []
        self.rewards = []
        self.radius_std = nn.Parameter(torch.tensor(1.0))
        self.angle_std = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, state):
        state = torch.FloatTensor(state)
        output = self.network(state)
        radius_mean = torch.sigmoid(output[0]) * 5.0 + 1e-6  # Scale to action space
        angle_mean = torch.sigmoid(output[1]) * 2 * torch.pi + 1e-6
        radius_dist = D.Normal(radius_mean, torch.clamp(torch.exp(self.radius_std), min=1e-6, max=2.0))
        angle_dist = D.Normal(angle_mean, torch.clamp(torch.exp(self.angle_std), min=1e-6, max=2.0))
        return radius_dist, angle_dist
    
    def select_action(self, state):
        radius_dist, angle_dist = self.forward(state)
        radius = radius_dist.sample()
        angle = angle_dist.sample()
        self.saved_actions.append((radius_dist.log_prob(radius), angle_dist.log_prob(angle)))
        return radius.item(), angle.item()
    
    def update(self, gamma=0.99):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if returns.std() > 1e-8:  # Avoid division by zero or very small numbers
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()  # Center the returns but skip scaling

        
        policy_loss = []
        for (radius_log_prob, angle_log_prob), R in zip(self.saved_actions, returns):
            policy_loss.append(-(radius_log_prob + angle_log_prob) * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.saved_actions = []
        self.rewards = []

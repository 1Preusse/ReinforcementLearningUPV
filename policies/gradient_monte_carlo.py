import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradientMonteCarloPolicy(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=10):
        super().__init__()
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [radius, angle]
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.01)
        
        self.saved_states = []
        self.saved_actions = []
        self.rewards = []
        
    def select_action(self, state, episode=0):
        state_tensor = torch.FloatTensor(state)
        output = self.policy_network(state_tensor)
        
        # Add exploration noise
        exploration_factor = max(0.1, 2.0 - episode/500)  # Decay from 1.0 to 0.1
        noise_radius = torch.randn(1) * 0.2 * exploration_factor
        noise_angle = torch.randn(1) * 1.0 * exploration_factor
        
        radius = 5.0 * torch.sigmoid(output[0])
        angle = 2 * np.pi * torch.sigmoid(output[1] + noise_angle)
        
        self.saved_states.append(state_tensor)
        self.saved_actions.append((radius, angle))
        
        return radius.item(), angle.item()
    
    def update(self, gamma=0.99):
        # Convert rewards to returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stable training
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient loss
        policy_loss = []
        for (radius, angle), R in zip(self.saved_actions, returns):
            # Maximize reward by minimizing negative log probability * return
            log_prob = torch.log(radius) + torch.log(angle)
            policy_loss.append(-log_prob * R)  # Negative because we want to maximize reward
        
        # Update policy
        policy_loss = torch.stack(policy_loss).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Clear episode memory
        self.saved_states = []
        self.saved_actions = []
        self.rewards = []

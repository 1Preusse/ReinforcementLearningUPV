import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt

class ContinuousGridworld:
    def __init__(self):
        self.start = np.array([0.5, 3.5])
        self.goal = np.array([7.5, 3.5])
        self.x_min, self.x_max = 0, 10
        self.y_min, self.y_max = 0, 8

    def reset(self):
        self.state = self.start.copy()
        return self.state

    def step(self, radius, angle):
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        next_state = self.state + np.array([dx, dy])

        # Clip to bounds
        next_state[0] = np.clip(next_state[0], self.x_min, self.x_max)
        next_state[1] = np.clip(next_state[1], self.y_min, self.y_max)

        # Reward
        distance_to_goal = np.linalg.norm(next_state - self.goal)
        reward = 100 if distance_to_goal < 0.5 else -1
        done = distance_to_goal < 0.5

        self.state = next_state
        return next_state, reward, done


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Mean outputs: radius and angle
        )
        self.radius_std = nn.Parameter(torch.tensor(1.0))
        self.angle_std = nn.Parameter(torch.tensor(1.0))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        state = torch.FloatTensor(state)
        output = self.network(state)
        radius_mean = torch.sigmoid(output[0]) * 5.0
        angle_mean = torch.sigmoid(output[1]) * 2 * torch.pi
        
        # Ensure standard deviation is always positive (clamped to a small value)
        radius_std = torch.clamp(torch.exp(self.radius_std), min=1e-6)
        angle_std = torch.clamp(torch.exp(self.angle_std), min=1e-6)
        
        radius_dist = D.Normal(radius_mean, radius_std)
        angle_dist = D.Normal(angle_mean, angle_std)
        
        return radius_dist, angle_dist

    def select_action(self, state):
        radius_dist, angle_dist = self.forward(state)
        radius = radius_dist.sample()
        angle = angle_dist.sample()
        return radius.item(), angle.item()

    def update(self, log_probs, rewards, gamma=0.99):
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        
        # Normalize rewards (avoid division by zero)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)

        loss = torch.stack(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_pg():
    env = ContinuousGridworld()
    policy = PolicyNetwork()
    num_episodes = 300
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs, rewards = [], []
        steps = 0

        while True:
            radius, angle = policy.select_action(state)
            next_state, reward, done = env.step(radius, angle)
            radius_dist, angle_dist = policy(state)
            log_probs.append(radius_dist.log_prob(torch.tensor(radius)) +
                             angle_dist.log_prob(torch.tensor(angle)))
            rewards.append(reward)
            state = next_state
            steps += 1
            if done:
                break

        policy.update(log_probs, rewards)
        steps_per_episode.append(steps)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Steps: {steps}")

    plt.plot(np.convolve(steps_per_episode, np.ones(50) / 50, mode="valid"))
    plt.xlabel("Episode")
    plt.ylabel("Average Steps")
    plt.title("Policy Gradient Performance Over Time")
    plt.show()


if __name__ == "__main__":
    train_pg()

import numpy as np
import matplotlib.pyplot as plt

class WindyGridworldSARSA:
    def __init__(self):
        self.x_min, self.x_max = 0, 10
        self.y_min, self.y_max = 0, 8
        self.start = np.array([0.5, 3.5])
        self.goal = np.array([7.5, 3.5])
        self.wind_strength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Wind off
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

    def reset(self):
        self.state = self.start.copy()
        return self.state

    def step(self, action):
        direction = self.actions[action]
        distance = 1  # No random noise for now
        next_state = self.state + np.array(direction) * distance

        # Clip to bounds
        next_state[0] = np.clip(next_state[0], self.x_min, self.x_max)
        next_state[1] = np.clip(next_state[1], self.y_min, self.y_max)

        # Reward
        distance_to_goal = np.linalg.norm(next_state - self.goal)
        if distance_to_goal < 0.8:
            print("OMG, we're done!")
            return next_state, 1000, True  # Large reward for reaching the goal
        else:
            return next_state, -0.1, False  # Small penalty for each step

class SemiGradientSARSA:
    def __init__(self, state_dim=2, action_dim=4, alpha=0.1, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.weights = np.random.randn((state_dim + 1) * action_dim) * 0.1

    def featurize(self, state, action):
        # Normalize state to a [0, 1] range for better convergence
        normalized_state = (state - np.array([5, 4])) / np.array([5, 4])
        state_features = np.concatenate([normalized_state, [1]])  # Add bias term
        action_one_hot = np.eye(self.action_dim)[action]  # One-hot encoding of action
        features = np.outer(state_features, action_one_hot).flatten()
        return np.nan_to_num(features)

    def eval(self, state, action):
        features = self.featurize(state, action)
        return np.dot(self.weights, features)

    def train(self, state, action, target):
        features = self.featurize(state, action)
        td_error = target - self.eval(state, action)
        update = self.alpha * td_error * features
        self.weights += update

    def select_action(self, state, eps):
        q_values = [self.eval(state, a) for a in range(self.action_dim)]

        # Epsilon-greedy policy
        if np.random.rand() > eps:
            return np.argmax(q_values)
        else:
            return np.random.choice(self.action_dim)


def ep_semi_grad_sarsa(env, ep, gamma, eps, q_hat, callback=None):
    """Episodic Semi-Gradient Sarsa"""
    
    def policy(st, q_hat, eps):
        q_values = [q_hat.eval(st, a) for a in range(len(env.actions))]  # Use env.actions length
        if np.random.rand() > eps:
            return np.argmax(q_values)  # Greedy action
        else:
            return np.random.choice(len(env.actions))  # Random action (epsilon-greedy)

    steps_per_episode = []

    for e_ in range(ep):
        S = env.reset()
        A = policy(S, q_hat, eps)
        steps = 0
        while True:
            S_, R, done = env.step(A)
            if done:
                q_hat.train(S, A, R)
                steps += 1
                break
            A_ = policy(S_, q_hat, eps)
            target = R + gamma * q_hat.eval(S_, A_)
            q_hat.train(S, A, target)
            S, A = S_, A_
            steps += 1
            print(steps)
            print(S)
        steps_per_episode.append(steps)

        if callback is not None:
            callback(e_, q_hat)

    return steps_per_episode


def train_sarsa():
    env = WindyGridworldSARSA()
    agent = SemiGradientSARSA()

    num_episodes = 1000  # Increase episodes for more learning
    epsilon = 1.0  # Start with full exploration

    # Train using episodic semi-gradient SARSA
    steps_per_episode = ep_semi_grad_sarsa(env, num_episodes, agent.gamma, epsilon, agent)

    # Smoothing
    window = max(1, min(50, len(steps_per_episode)))
    plt.plot(np.convolve(steps_per_episode, np.ones(window) / window, mode="valid"))
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("SARSA Performance Over Time")
    plt.show()

if __name__ == "__main__":
    train_sarsa()

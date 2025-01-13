import numpy as np
from .base_policy import BasePolicy

class GradientDescentPolicy(BasePolicy):
    def __init__(self, state_space, action_space, learning_rate=0.01):
        """
        gradient descent policy

        Parameters:
        - state_space: Number of state features (input dimensions).
        - action_space: Number of possible actions (output dimensions).
        - learning_rate: Step size for parameter updates.
        """
        super.__init__(action_space)
        self.state_space = state_space
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = np.random.randn(state_space, action_space) * 0.01
        self.bias = np.zeros(action_space)

        # Placeholder for rewards and gradients
        self.rewards = []
        self.gradients = []

    def softmax(self, logits):
        """
        Compute the softmax of logits to get action probabilities.
        """
        exp_logits = np.exp(logits - np.max(logits))  # Avoid numerical overflow
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state):
        """
        Select an action based on the current policy.

        Parameters:
        - state: Current state (numpy array of shape (state_space,)).

        Returns:
        - action: Selected action.
        - log_prob: Log-probability of the chosen action.
        """
        logits = np.dot(state, self.weights) + self.bias
        probabilities = self.softmax(logits)

        # Sample an action using the probabilities
        action = np.random.choice(self.action_space, p=probabilities)
        log_prob = np.log(probabilities[action])

        # Store the gradient of the log-probability w.r.t. weights and bias
        grad_logits = -probabilities
        grad_logits[action] += 1  # ∂log(pi(a|s)) / ∂logits = 1 - pi(a|s)
        self.gradients.append((state, grad_logits))

        return action, log_prob

    def store_reward(self, reward):
        """
        Store the reward for the current step.
        """
        self.rewards.append(reward)

    def compute_discounted_rewards(self, discount_factor):
        """
        Compute the discounted rewards for all steps in an episode.

        Parameters:
        - discount_factor: Discount factor (gamma) for future rewards.

        Returns:
        - discounted_rewards: Numpy array of discounted rewards.
        """
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(self.rewards))):
            cumulative = cumulative * discount_factor + self.rewards[t]
            discounted_rewards[t] = cumulative

        # Normalize the rewards for stability
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-8

        return discounted_rewards

    def update_policy(self, discount_factor=0.99):
        """
        Update the policy parameters using gradient descent.

        Parameters:
        - discount_factor: Discount factor (gamma) for future rewards.
        """
        discounted_rewards = self.compute_discounted_rewards(discount_factor)

        # Aggregate gradients for weights and bias
        total_weight_grad = np.zeros_like(self.weights)
        total_bias_grad = np.zeros_like(self.bias)

        for (state, grad_logits), reward in zip(self.gradients, discounted_rewards):
            grad_weights = np.outer(state, grad_logits)  # Gradients w.r.t. weights
            grad_bias = grad_logits  # Gradients w.r.t. bias

            total_weight_grad += grad_weights * reward
            total_bias_grad += grad_bias * reward

        # Perform gradient descent
        self.weights += self.learning_rate * total_weight_grad
        self.bias += self.learning_rate * total_bias_grad

        # Clear rewards and gradients after update
        self.rewards = []
        self.gradients = []


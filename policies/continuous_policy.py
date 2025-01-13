import numpy as np
from .base_policy import BasePolicy

class ContinuousPolicy(BasePolicy):
    def __init__(self, action_space, state_dim=2, action_dim=4):
        super().__init__(action_space)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = np.random.randn(self.state_dim, self.action_dim)
        self.learning_rate = 0.01
    
    def select_action(self, state):
        logits = np.dot(state, self.weights)
        discrete_action = np.argmax(logits)
        return discrete_action
    
    def compute_policy_gradient(self, state, action, reward):
        # Compute softmax probabilities
        logits = np.dot(state, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Create one-hot encoding of the action
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1
        
        # Compute gradient
        gradient = np.outer(state, (action_one_hot - probs) * reward)
        return gradient
    
    def update(self, state, action, reward, next_state):
        gradient = self.compute_policy_gradient(state, action, reward)
        self.weights += self.learning_rate * gradient
        self.weights += self.learning_rate * gradient
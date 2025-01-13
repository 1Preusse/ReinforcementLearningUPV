import random
from .base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, action_space, epsilon=0.1):
        """
        initialize the epsilon-greedy policy

        Parameters:
        - action_space: The set of possible actions.
        - epsilon (float): Probability of choosing a random action.
        """
        super().__init__(action_space)
        self.epsilon = epsilon

    def select_action(self, state, values):
        """
        select action with epsilon-greedy policy

        Parameters:
        - state: The current state of the environment
        - values: Value estimates for actions (Q)

        Returns:
        - action: The selected actoin
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return max(range(len(values)), key=lambda i: values[i])
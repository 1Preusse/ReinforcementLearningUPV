from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, action_space):
        """
        constructor of base policy

        Parameters:
        - action_space: The set of possible actions the policy can select from.
        """
        self.action_space = action_space

    @abstractmethod
    def select_action(self, state, values=None):
        """
        abstract method to show what the select_action function should be

        Parameters:
        - state: The current state of the environment.
        - values: Optional, value estimates for actions (e.g., Q-values).

        Returns:
        - action: The selected action.
        """
        pass

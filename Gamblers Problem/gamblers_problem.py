import numpy as np
import matplotlib.pyplot as plt

# Parameters
goal = 100
p_head = 0.4  # Probability of heads
gamma = 1.0  # Discount factor for value iteration (can be 1 for this problem)

# Initialize value function and states
value_function = np.zeros(goal + 1)
value_function[goal] = 1  # The goal state has a reward of 1
states = np.arange(1, goal)

# Function to get possible actions for a given state
def possible_actions(capital):
    return np.arange(1, min(capital, goal - capital) + 1)

# Value Iteration
def value_iteration(value_function, p_head, theta=1e-6):
    delta = float('inf')
    while delta > theta:
        delta = 0
        # Loop over all states except the terminal ones (0 and goal)
        for state in states:
            old_value = value_function[state]
            action_values = []
            # Loop over all possible actions for the current state
            for action in possible_actions(state):
                win_state = state + action
                lose_state = state - action
                # Calculate the expected value for taking the current action
                action_value = (
                    p_head * (value_function[win_state] if win_state <= goal else 0) +
                    (1 - p_head) * (value_function[lose_state] if lose_state >= 0 else 0)
                )
                action_values.append(action_value)
            # Update the value function with the maximum value of all actions
            value_function[state] = max(action_values)
            # Update the delta to check for convergence
            delta = max(delta, abs(old_value - value_function[state]))
    return value_function

# Function to calculate the optimal policy
def calculate_optimal_policy(value_function, p_head):
    policy = np.zeros(goal + 1)
    for state in states:
        action_values = []
        for action in possible_actions(state):
            win_state = state + action
            lose_state = state - action
            action_value = (
                p_head * (value_function[win_state] if win_state <= goal else 0) +
                (1 - p_head) * (value_function[lose_state] if lose_state >= 0 else 0)
            )
            action_values.append(action_value)
        # Choose the action that yields the highest value
        policy[state] = possible_actions(state)[np.argmax(action_values)]
    return policy

# Run value iteration
value_function = value_iteration(value_function, p_head)

# Calculate the optimal policy
optimal_policy = calculate_optimal_policy(value_function, p_head)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(value_function)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.title('Value Function')

plt.subplot(1, 2, 2)
plt.plot(optimal_policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.title('Optimal Policy')
plt.tight_layout()
plt.show()
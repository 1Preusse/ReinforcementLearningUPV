import numpy as np
import random
import matplotlib.pyplot as plt

random_seed = 42
np.random.seed(random_seed)



def create_wind_grid(rows=7, cols=10):
    grid = np.zeros((rows, cols))
    grid[:, 3:6] = 1
    grid[:, 6:8] = 2
    grid[:, 8] = 1
    return grid

def get_next_state(current_state, action, wind_grid):
    row, col = current_state
    wind = int(wind_grid[row, col])  # Ensure wind is an integer
    
    # Define actions: 0: up, 1: right, 2: down, 3: left
    if action == 0:  # up
        row = max(row - 1 - wind, 0)
    elif action == 1:  # right
        col = min(col + 1, wind_grid.shape[1] - 1)
        row = max(row - wind, 0)
    elif action == 2:  # down
        row = min(max(row + 1 - wind, 0), wind_grid.shape[0] - 1)
    elif action == 3:  # left
        col = max(col - 1, 0)
        row = max(row - wind, 0)
    
    return (int(row), int(col))  # Ensure returned values are integers

def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return int(np.argmax(Q[state[0], state[1]]))

def sarsa(grid, start, goal, episodes, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((grid.shape[0], grid.shape[1], 4))
    episode_steps = []
    total_steps = 0
    
    for _ in range(episodes):
        state = start
        action = epsilon_greedy_policy(Q, state, epsilon)
        steps = 0
        
        while state != goal:
            next_state = get_next_state(state, action, grid)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            
            reward = -1  # -1 reward for each step
            
            # SARSA update
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            
            state = next_state
            action = next_action
            steps += 1
            total_steps += 1
            if state == (3,1):
                print(Q[state[0], state[1]])
        
        episode_steps.append((total_steps, steps))
    
    return Q, episode_steps



###!!!!!THIS IS BROKEN BUT I HAVE NOT FOUND THE MISTAKE YET!!!!!###
def sarsa_lambda(grid, start, goal, episodes, alpha=0.5, gamma=1.0, epsilon=0.1, lambda_=0.9):
    """
    SARSA(λ) algorithm with eligibility traces.
    
    Args:
        grid: The windy gridworld environment
        start: Starting position tuple (row, col)
        goal: Goal position tuple (row, col)
        episodes: Number of episodes to run
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy policy
        lambda_: Eligibility trace decay rate
    """
    Q = np.zeros((grid.shape[0], grid.shape[1], 4))
    episode_steps = []
    total_steps = 0
    
    for episode in range(episodes):
        state = start
        action = epsilon_greedy_policy(Q, state, epsilon)
        steps = 0
        
        # Initialize eligibility traces for this episode
        E = np.zeros_like(Q)
        
        while state != goal:
            next_state = get_next_state(state, action, grid)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            reward = -1
            
            # Calculate TD error
            delta = (reward + 
                    gamma * Q[next_state[0], next_state[1], next_action] - 
                    Q[state[0], state[1], action])
            
            # Accumulating traces
            E[state[0], state[1], action] = min(E[state[0], state[1], action] + 1, 1.0)
            
            # Update Q-values for all states
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    for a in range(4):
                        Q[i, j, a] += alpha * delta * E[i, j, a]
            
            # Decay eligibility traces
            E *= gamma * lambda_
            
            state = next_state
            action = next_action
            steps += 1
            total_steps += 1
            
            # Add a maximum step limit to prevent infinite loops during training
            if steps > 1000:  # Maximum steps per episode
                break
                
        episode_steps.append((total_steps, steps))
        
        # Print progress periodically
        if episode % 10 == 0:
            print(f"Episode {episode}, Steps: {steps}")
    
    return Q, episode_steps

def double_sarsa(grid, start, goal, episodes, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Double SARSA algorithm to reduce overestimation bias.
    
    Args:
        grid: The windy gridworld environment
        start: Starting position tuple (row, col)
        goal: Goal position tuple (row, col)
        episodes: Number of episodes to run
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy policy
    
    Returns:
        Q: Average of Q1 and Q2 (final action-value function)
        episode_steps: List of (total_steps, steps_this_episode) tuples
    """
    # Initialize two sets of Q-values
    Q1 = np.zeros((grid.shape[0], grid.shape[1], 4))
    Q2 = np.zeros((grid.shape[0], grid.shape[1], 4))
    episode_steps = []
    total_steps = 0
    
    for _ in range(episodes):
        state = start
        # Choose action based on the average of both Q-values
        action = epsilon_greedy_policy((Q1 + Q2)/2, state, epsilon)
        steps = 0
        
        while state != goal:
            next_state = get_next_state(state, action, grid)
            next_action = epsilon_greedy_policy((Q1 + Q2)/2, next_state, epsilon)
            reward = -1
            
            # Randomly update either Q1 or Q2
            if np.random.random() < 0.5:
                Q1[state[0], state[1], action] += alpha * (
                    reward + gamma * Q2[next_state[0], next_state[1], next_action] - 
                    Q1[state[0], state[1], action]
                )
            else:
                Q2[state[0], state[1], action] += alpha * (
                    reward + gamma * Q1[next_state[0], next_state[1], next_action] - 
                    Q2[state[0], state[1], action]
                )
            
            state = next_state
            action = next_action
            steps += 1
            total_steps += 1
        
        episode_steps.append((total_steps, steps))
    
    # Return the average of both Q-value estimates
    return (Q1 + Q2)/2, episode_steps

def plot_grid(grid, Q, path=None, start=None, goal=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the grid
    im = ax.imshow(grid, cmap='YlGnBu')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Wind strength', rotation=270, labelpad=15)

    # Add grid lines
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add policy arrows
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            action = np.argmax(Q[i, j])
            if action == 0:  # up
                ax.arrow(j, i, 0, -0.4, head_width=0.3, head_length=0.3, fc='k', ec='k')
            elif action == 1:  # right
                ax.arrow(j, i, 0.4, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')
            elif action == 2:  # down
                ax.arrow(j, i, 0, 0.4, head_width=0.3, head_length=0.3, fc='k', ec='k')
            elif action == 3:  # left
                ax.arrow(j, i, -0.4, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')

    # Plot path if provided
    if path is not None:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=8)

    # Mark start and goal positions
    if start is not None:
        ax.text(start[1], start[0], "S", ha='center', va='center', color='white', fontweight='bold', fontsize=20)
    if goal is not None:
        ax.text(goal[1], goal[0], "G", ha='center', va='center', color='white', fontweight='bold', fontsize=20)

    # Add labels and title
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_title('Windy Gridworld - Learned Policy and Optimal Path')

    plt.tight_layout()
    plt.show()

def plot_episode_steps(episode_steps):
    episodes = range(len(episode_steps))
    total_steps = [step[0] for step in episode_steps]
    steps_per_episode = [step[1] for step in episode_steps]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Total Timesteps')
    ax1.set_ylabel('Episodes', color=color)
    ax1.plot(total_steps, episodes, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Timesteps per Episode', color=color)
    ax2.plot(total_steps, steps_per_episode, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Episodes and Timesteps per Episode over Total Timesteps')
    fig.tight_layout()
    plt.show()

def print_state_values(Q, state):
    """Print Q-values for all actions at a given state."""
    actions = ['Up', 'Right', 'Down', 'Left']
    print(f"\nQ-values at state {state}:")
    for a, action_name in enumerate(actions):
        print(f"{action_name}: {Q[state[0], state[1], a]:.3f}")

# Initialize the environment
grid = create_wind_grid()
start = (3, 0)
goal = (3, 7)

# Run SARSA
Q, episode_steps = sarsa_lambda(grid, start, goal, 
                              episodes=170,
                              alpha=0.1,      # Reduced learning rate
                              gamma=0.9,      # Slightly reduced discount factor
                              epsilon=0.1,    # Keep exploration rate
                              lambda_=0.7)    # Reduced trace decay rate

# Plot episode steps
plot_episode_steps(episode_steps)

# Test the learned policy
current_state = start
path = [current_state]

while current_state != goal:
    action = int(np.argmax(Q[current_state[0], current_state[1]]))
    print(action)
    next_state = get_next_state(current_state, action, grid)
    path.append(next_state)
    current_state = next_state
    #print(f"Action: {action}, New state: {current_state}")
    

print("Goal reached!")
print("Final path:", path)

# Visualize the results
plot_grid(grid, Q, path=path, start=start, goal=goal)

# After training, check Q-values at specific states
print_state_values(Q, start)
print_state_values(Q, (3, 1))  # Check any state where the policy seems to loop
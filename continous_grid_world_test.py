import numpy as np
import matplotlib.pyplot as plt
from environments.continuous_grid_world_environment import ContinuousWindyGridworld
from policies.sgd_policy import SGDPolicy

# Training parameters
num_episodes = 1000
num_trials = 50
gamma = 0.99

# Initialize storage for average steps
all_steps = np.zeros((num_trials, num_episodes))

# Perform multiple trials
for trial in range(num_trials):
    env = ContinuousWindyGridworld()
    policy = SGDPolicy()
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_steps = 0
        policy.rewards = []
        
        while not done:
            action = policy.select_action(state)
            next_state, reward, done = env.step(action)
            policy.rewards.append(reward)
            state = next_state
            episode_steps += 1
            
            if done:
                policy.update(gamma)
        
        all_steps[trial, episode] = episode_steps

        if (trial == 0) and ((episode + 1) % 100 == 0):  # Print for the first trial
            print(f"Trial {trial + 1}, Episode {episode + 1}/{num_episodes}, Steps: {episode_steps}")

# Compute average steps over all trials
avg_steps_per_episode = all_steps.mean(axis=0)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(avg_steps_per_episode, label="Average Steps (50 Runs)")
plt.xlabel("Episode")
plt.ylabel("Average Steps to Goal")
plt.title("Policy Improvement Over Multiple Trials")
plt.legend()
plt.grid()
plt.show()


# Visualize final policy
state = env.reset()
path = [state]
done = False

while not done:
    action = policy.select_action(state)
    next_state, _, done = env.step(action)
    path.append(next_state)
    state = next_state

path = np.array(path)
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], marker='o')
plt.scatter(env.goal[0], env.goal[1], c='red', label='Goal')
plt.xlim(env.x_min, env.x_max)
plt.ylim(env.y_min, env.y_max)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Final Policy Path")
plt.legend()
plt.grid()
plt.show()

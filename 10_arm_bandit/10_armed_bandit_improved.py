import numpy as np
import matplotlib.pyplot as plt

def greedy_choice(Q):
    return np.argmax(Q)

def epsilon_greedy_choice(Q, epsilon, n_arms):
    if np.random.random() < epsilon:
        return np.random.choice(n_arms)
    else:
        return np.argmax(Q)

def run_bandit_simulation(n_arms, n_runs, n_steps, epsilon, seed=18798312, alternative=False):
    np.random.seed(seed)
    
    results = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        n_arms_mean = np.random.normal(0, 1, n_arms)
        
        for step in range(n_steps):

            # every 300 steps take a epsilon greedy approach for 20 steps.
            # otherwise select greedy action
            if alternative:   
                if step%300<20:
                    action = epsilon_greedy_choice(Q,0.2,n_arms)
                else:
                    action = epsilon_greedy_choice(Q,0,n_arms)
            else:
                action = epsilon_greedy_choice(Q, epsilon, n_arms)
            reward = np.random.normal(n_arms_mean[action], 1)
            
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            
            results[run, step] = reward
    
    return results

# Set up parameters
n_arms = 10
n_runs = 2000
n_steps = 2000

# Run simulations
greedy_results = run_bandit_simulation(n_arms, n_runs, n_steps, 0)
epsilon_01_results = run_bandit_simulation(n_arms, n_runs, n_steps, 0.1)
epsilon_001_results = run_bandit_simulation(n_arms, n_runs, n_steps, 0.01,alternative=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.mean(greedy_results, axis=0), label='Greedy')
plt.plot(np.mean(epsilon_01_results, axis=0), label='ε-greedy (ε=0.1)')
plt.plot(np.mean(epsilon_001_results, axis=0), label='ε-greedy (ε=0.01)')

plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Multi-Armed Bandit: Greedy vs ε-greedy')
plt.legend()
plt.grid(True)
plt.show()
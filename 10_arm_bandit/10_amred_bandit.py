import numpy as np
import matplotlib.pyplot as plt
seed = 1879831231231
rng = np.random.default_rng(seed)

n_arms = 10
std_bandits = 1
mean_bandits = 0
std_rewards = 1
runs = 2000
steps = 1000





actions = np.arange(10)



# greedy 
# expect = 0, choose best, 

"""class greedy_bandit(n_arms):
    current_choice = np-random.randint(n_arms)
    expected_reward = np.zeros(n_arms)

    def choice(self,last_reward):
        if last_reward<np.argmax(expected_reward):
            self.current_choice = np.argmax(expected_reward)
        return self.current_choice"""
def greedy_choice(a,Q):
    a = np.argmax(Q)
    return a


def epsilon_greedy_choice(a,Q,epsilon):
    if rng.random() < epsilon:
        a = rng.choice(actions)
    else:
        a = np.argmax(Q)
    return a
data = np.zeros((runs, steps))
data_epsilon_greedy = np.zeros((runs, steps))
data_epsilon_greedy_01 = np.zeros((runs, steps))

for i in range(1,runs):
    rng = np.random.default_rng(seed + i)
    Q = np.ones(n_arms) * mean_bandits
    a = rng.choice(actions)
    q = rng.normal(mean_bandits, std_bandits,n_arms)
    R = rng.normal(q[a], std_rewards)
    # generate one run

    n_arm_numbers = rng.normal(q[:, np.newaxis], std_rewards, (n_arms, steps))
    
    for j in range(1,steps):
        reward = n_arm_numbers[a, j]
        Q[a] = (Q[a]) + (1/j)*(reward-Q[a])
        # make choice
        a = epsilon_greedy_choice(a,Q,0.01)
        data[i,j] = reward
        # generate one run
    for j in range(1,steps):
        reward = n_arm_numbers[a, j]
        Q[a] = (Q[a]) + (1/j)*(reward-Q[a])
        # make choice
        a = epsilon_greedy_choice(a,Q,0.3)
        data_epsilon_greedy[i,j] = reward
        # generate one run
    for j in range(1,steps):
        reward = n_arm_numbers[a, j]
        Q[a] = (Q[a]) + (1/j)*(reward-Q[a])
        # make choice
        a = epsilon_greedy_choice(a,Q,0.01)
        data_epsilon_greedy_01[i,j] = reward
        # generate one run



plt.plot(np.mean(data, axis=0))
plt.plot(np.mean(data_epsilon_greedy, axis=0))
plt.plot(np.mean(data_epsilon_greedy_01, axis=0))
plt.legend(["greedy","epsilon greedy 0.01", "epsilon greedy 0.1"])
plt.show()

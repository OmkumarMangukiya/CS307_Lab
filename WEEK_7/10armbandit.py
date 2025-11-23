import numpy as np

class NonStationaryBandit:
    def __init__(self, k=10, std_walk=0.01, init_mean=0): 
        self.k = k  
        self.means = np.full(k, float(init_mean))  
        self.std_walk = std_walk  
        self.time_step = 0  
        
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)

    def update_means(self):
        self.means += np.random.normal(0, self.std_walk, self.k)

    def pull(self, action):
        self.update_means()

        reward = self.means[action]
        return reward

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.k)
        else:
            action = np.argmax(self.q_estimates)
        
        return action

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

def simulate_bandit(n_steps=1000, epsilon=0.1):
    bandit = NonStationaryBandit()
    
    rewards = np.zeros(n_steps)

    for t in range(n_steps):
        action = bandit.select_action(epsilon)

        reward = bandit.pull(action)
        rewards[t] = reward

        bandit.update_estimates(action, reward)

    average_reward = np.mean(rewards)
    
    return rewards, average_reward

n_steps = 10000
epsilon = 0.1  
rewards, average_reward = simulate_bandit(n_steps, epsilon)

print(f"Average reward over {n_steps} steps: {average_reward}")

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Rewards over time for a 10-armed non-stationary bandit')
plt.show()
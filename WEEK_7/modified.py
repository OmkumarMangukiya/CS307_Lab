import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def _init_(self, k=10):
        self.k = k 
        self.true_values = np.zeros(k)
        self.optimal_action = 0
        self.time_step = 0
        
    def reset(self):
        self.true_values = np.zeros(self.k)
        self.time_step = 0
        
    def pull(self, action):
        self.time_step += 1
        self.true_values += np.random.normal(0, 0.01, self.k)
        
        reward = np.random.normal(self.true_values[action], 1)
        
        self.optimal_action = np.argmax(self.true_values)
        
        return reward
    
class ModifiedEpsilonGreedyAgent:
    def _init_(self, n_arms, epsilon=0.1, alpha=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha  
        self.q_values = np.zeros(n_arms)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_experiment(n_steps=10000, n_runs=100):
    bandit = NonStationaryBandit()
    agent = ModifiedEpsilonGreedyAgent(bandit.k)
    
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)
    
    for run in range(n_runs):
        bandit.reset()
        agent.q_values = np.zeros(bandit.k)
        
        for step in range(n_steps):
            action = agent.select_action()
            
            reward = bandit.pull(action)
            agent.update(action, reward)
            
            rewards[step] += reward
            optimal_actions[step] += (action == bandit.optimal_action)
    
    rewards /= n_runs
    optimal_actions /= n_runs
    
    return rewards, optimal_actions

n_steps = 10000
rewards, optimal_actions = run_experiment(n_steps)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')

plt.subplot(1, 2, 2)
plt.plot(optimal_actions)
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Optimal Action Selection Percentage')

plt.tight_layout()
plt.show()

print(f"Average reward over last 1000 steps: {np.mean(rewards[-1000:]):.3f}")
print(f"Optimal action percentage over last 1000 steps: {np.mean(optimal_actions[-1000:]) * 100:.1f}%")
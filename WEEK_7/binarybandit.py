import numpy as np
import matplotlib.pyplot as plt


def binary_bandit_a(action):
    p = [0.1, 0.2]  
    if np.random.rand() < p[action - 1]:
        return 1  
    else:
        return 0  


def binary_bandit_b(action):
    p = [0.8, 0.9]  
    if np.random.rand() < p[action - 1]:
        return 1  
    else:
        return 0  



def epsilon_greedy_bandit(epsilon, num_iterations, bandit_function):
    Q = [0, 0]  
    N = [0, 0]  
    total_reward = 0
    history_q1 = []
    history_q2 = []

    for t in range(num_iterations):
        if np.random.rand() < epsilon:
            action = np.random.choice([1, 2])   
        else:
            action = np.argmax(Q) + 1  

        reward = bandit_function(action)

        N[action - 1] += 1
        Q[action - 1] += (reward - Q[action - 1]) / N[action - 1]

        total_reward += reward
        
        if t % 10 == 0:
            history_q1.append(Q[0])
            history_q2.append(Q[1])

    print(f"\nFinal Action-Value Estimates: Q(1) = {Q[0]:.2f}, Q(2) = {Q[1]:.2f}")
    return history_q1, history_q2


print("Training Bandit A...")
history_a_q1, history_a_q2 = epsilon_greedy_bandit(0.1, 10000, binary_bandit_a)

print("Training Bandit B...")
history_b_q1, history_b_q2 = epsilon_greedy_bandit(0.1, 10000, binary_bandit_b)

iterations = np.arange(0, 10000, 10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(iterations, history_a_q1, label='Action 1 (p=0.1)', linewidth=2)
axes[0].plot(iterations, history_a_q2, label='Action 2 (p=0.2)', linewidth=2)
axes[0].axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='True value for Action 1')
axes[0].axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='True value for Action 2')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Action-Value Estimate (Q)')
axes[0].set_title('Bandit A: Low Success Rates')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(iterations, history_b_q1, label='Action 1 (p=0.8)', linewidth=2)
axes[1].plot(iterations, history_b_q2, label='Action 2 (p=0.9)', linewidth=2)
axes[1].axhline(y=0.8, color='blue', linestyle='--', alpha=0.5, label='True value for Action 1')
axes[1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='True value for Action 2')
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Action-Value Estimate (Q)')
axes[1].set_title('Bandit B: High Success Rates')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
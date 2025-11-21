import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("PROBLEM 1: ERROR CORRECTING CAPABILITY")
print("="*60)

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=50):
        state = np.array(pattern)
        for _ in range(max_iterations):
            new_state = np.sign(np.dot(self.weights, state))
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state

num_neurons = 100
hopfield = HopfieldNetwork(num_neurons)

num_patterns = 5
patterns = [np.random.choice([-1, 1], size=(num_neurons,)) for _ in range(num_patterns)]
hopfield.train(patterns)

test_pattern = patterns[0].copy()

noise_levels = [5, 10, 15, 20, 25]
print(f"\n{'Noise Level (%)':20} {'Bits Flipped':20} {'Successfully Recovered':20}")
print("-"*60)

for noise_pct in noise_levels:
    noisy_pattern = test_pattern.copy()
    num_flips = num_neurons * noise_pct // 100
    noise_indices = np.random.choice(num_neurons, size=num_flips, replace=False)
    noisy_pattern[noise_indices] *= -1
    
    recalled_pattern = hopfield.recall(noisy_pattern)
    is_recovered = np.array_equal(recalled_pattern, test_pattern)
    print(f"{noise_pct}%{20} {num_flips}{20} {is_recovered}")

print(f"\nTheoretical capacity: {int(0.15 * num_neurons)} patterns")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(test_pattern.reshape(10, 10), cmap='gray')
axes[0].set_title('Original Pattern')
axes[0].set_ylabel('Row')
axes[0].set_xlabel('Column')

noisy_pattern_10 = test_pattern.copy()
noise_indices = np.random.choice(num_neurons, size=10, replace=False)
noisy_pattern_10[noise_indices] *= -1
axes[1].imshow(noisy_pattern_10.reshape(10, 10), cmap='gray')
axes[1].set_title('Noisy Pattern (10% noise)')
axes[1].set_ylabel('Row')
axes[1].set_xlabel('Column')

recalled = hopfield.recall(noisy_pattern_10)
axes[2].imshow(recalled.reshape(10, 10), cmap='gray')
axes[2].set_title('Recalled Pattern')
axes[2].set_ylabel('Row')
axes[2].set_xlabel('Column')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("PROBLEM 2: EIGHT-ROOK PROBLEM")
print("="*60)

n_rooks = 8
n_units = n_rooks * n_rooks

W_rook = np.zeros((n_units, n_units))
gamma_rook = 100

def get_index_rook(row, col, n):
    return row * n + col

for i in range(n_rooks):
    for j in range(n_rooks):
        for k in range(n_rooks):
            if j != k:
                idx1 = get_index_rook(i, j, n_rooks)
                idx2 = get_index_rook(i, k, n_rooks)
                W_rook[idx1, idx2] -= gamma_rook
                
            if i != k:
                idx1 = get_index_rook(i, j, n_rooks)
                idx2 = get_index_rook(k, j, n_rooks)
                W_rook[idx1, idx2] -= gamma_rook

np.fill_diagonal(W_rook, 0)

print(f"\nBoard size: {n_rooks}x{n_rooks}")
print(f"Number of neurons: {n_units}")
print(f"Number of weights: {n_units * n_units}")
print(f"Penalty parameter (gamma): {gamma_rook}")
print(f"Non-zero weights: {np.count_nonzero(W_rook)}")
print(f"\nWeight reason: Negative weights penalize two rooks in same row/column")

for attempt in range(20):
    state_rook = np.zeros((n_rooks, n_rooks))
    perm = np.random.permutation(n_rooks)
    for i in range(n_rooks):
        state_rook[i, perm[i]] = 1
    
    for iteration in range(100):
        prev_state = state_rook.copy()
        for i in range(n_rooks):
            for j in range(n_rooks):
                idx = get_index_rook(i, j, n_rooks)
                input_sum = np.dot(W_rook[idx, :], state_rook.flatten())
                state_rook[i, j] = 1 if input_sum >= 0 else 0
        
        if np.array_equal(state_rook, prev_state):
            break
    
    row_sums = np.sum(state_rook, axis=1)
    col_sums = np.sum(state_rook, axis=0)
    
    if np.all(row_sums == 1) and np.all(col_sums == 1):
        print(f"\nValid solution found in attempt {attempt + 1}:")
        print(state_rook.astype(int))
        rook_solution = state_rook.copy()
        break

plt.figure(figsize=(8, 8))
for i in range(n_rooks):
    for j in range(n_rooks):
        if (i + j) % 2 == 0:
            color = 'lightgray'
        else:
            color = 'black'
        plt.fill_between([j, j+1], i, i+1, color=color)

for i in range(n_rooks):
    for j in range(n_rooks):
        if rook_solution[i, j] == 1:
            plt.plot(j+0.5, i+0.5, marker='s', markersize=20, color='white', markeredgecolor='red', markeredgewidth=2)

plt.xlim(0, n_rooks)
plt.ylim(0, n_rooks)
plt.gca().invert_yaxis()

column_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
row_labels = ['8', '7', '6', '5', '4', '3', '2', '1']

plt.xticks(range(n_rooks + 1), [''] + column_labels, fontsize=12)
plt.yticks(range(n_rooks + 1), [''] + row_labels, fontsize=12)

plt.grid(True, color='white', linewidth=2)
plt.title('Eight-Rook Problem Solution', fontsize=14)
plt.xlabel('Column')
plt.ylabel('Row')
plt.show()

print("\n" + "="*60)
print("PROBLEM 3: TSP WITH 10 CITIES")
print("="*60)

n_cities = 10
gamma = 1000
city_coordinates = np.random.rand(n_cities, 2) * 100

d = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        d[i, j] = np.linalg.norm(city_coordinates[i] - city_coordinates[j])

print(f"\nNumber of cities: {n_cities}")
print(f"Number of neurons: {n_cities * n_cities}")
print(f"Number of weights: {(n_cities * n_cities) ** 2}")
print(f"Penalty parameter (gamma): {gamma}")
print(f"\nWeights explanation:")
print(f"  1. Distance weights: Penalize long paths")
print(f"  2. Row constraint: Each city appears once")
print(f"  3. Column constraint: One city per position")

n_units_tsp = n_cities * n_cities
W_tsp = np.zeros((n_units_tsp, n_units_tsp))
biases_tsp = -gamma / 2 * np.ones(n_units_tsp)

def get_index_tsp(city, position, n):
    return city * n + position

for i in range(n_cities):
    for k in range(n_cities):
        for j in range(n_cities):
            for l in range(n_cities):
                if k == (l - 1) % n_cities:
                    W_tsp[get_index_tsp(i, k, n_cities), get_index_tsp(j, l, n_cities)] -= d[i, j]

for i in range(n_cities):
    for k in range(n_cities):
        for l in range(n_cities):
            if k != l:
                W_tsp[get_index_tsp(i, k, n_cities), get_index_tsp(i, l, n_cities)] -= gamma
        for j in range(n_cities):
            if i != j:
                W_tsp[get_index_tsp(i, k, n_cities), get_index_tsp(j, k, n_cities)] -= gamma

np.fill_diagonal(W_tsp, 0)

def calculate_total_distance(tour, dist_matrix):
    total = 0
    for i in range(len(tour) - 1):
        total += dist_matrix[tour[i], tour[i + 1]]
    return total

best_tour_tsp = None
best_distance_tsp = float('inf')

for repeat in range(50):
    state_tsp = np.zeros((n_cities, n_cities))
    perm = np.random.permutation(n_cities)
    for k in range(n_cities):
        state_tsp[perm[k], k] = 1
    
    for iteration in range(1000):
        prev_state = state_tsp.copy()
        indices = [(i, k) for i in range(n_cities) for k in range(n_cities)]
        np.random.shuffle(indices)
        
        for i, k in indices:
            idx = get_index_tsp(i, k, n_cities)
            input_sum = np.dot(W_tsp[idx, :], state_tsp.flatten()) + biases_tsp[idx]
            state_tsp[i, k] = 1 if input_sum >= -50 else 0
        
        if np.array_equal(state_tsp, prev_state):
            break
    
    tour = []
    used_cities = set()
    
    for step in range(n_cities):
        found = False
        for city in range(n_cities):
            if state_tsp[city, step] == 1 and city not in used_cities:
                tour.append(city)
                used_cities.add(city)
                found = True
                break
        
        if not found:
            for city in range(n_cities):
                if city not in used_cities:
                    tour.append(city)
                    used_cities.add(city)
                    break
    
    if len(tour) == n_cities:
        tour.append(tour[0])
        total_dist = calculate_total_distance(tour, d)
        
        if total_dist < best_distance_tsp:
            best_distance_tsp = total_dist
            best_tour_tsp = tour

print(f"\nBest tour found: {best_tour_tsp}")
print(f"Best distance: {best_distance_tsp:.2f}")

if best_tour_tsp is not None:
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(city_coordinates):
        plt.scatter(x, y, color="red", s=200, zorder=5)
        plt.text(x + 2, y + 2, f"{i}", fontsize=12, fontweight='bold')

    for i in range(len(best_tour_tsp) - 1):
        city_a = city_coordinates[best_tour_tsp[i]]
        city_b = city_coordinates[best_tour_tsp[i + 1]]
        plt.plot([city_a[0], city_b[0]], [city_a[1], city_b[1]], "b-", linewidth=2)

    plt.title(f"TSP Solution (10 Cities) - Distance: {best_distance_tsp:.2f}", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.show()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Problem 1: Error correction - Network recovers patterns with noise")
print("Problem 2: 8-Rook problem - Valid solution with constraint satisfaction")
print("Problem 3: TSP - Optimal tour for 10 cities using Hopfield network")
print(f"TSP weights required: {n_units_tsp * n_units_tsp} = {(n_cities*n_cities)**2}")
print("="*60)
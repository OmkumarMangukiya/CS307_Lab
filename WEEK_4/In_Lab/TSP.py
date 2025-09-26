import numpy as np
import random
import matplotlib.pyplot as plt

locations = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Ajmer": (26.4499, 74.6399),
    "Jaisalmer": (26.9157, 70.9083),
    "Bikaner": (28.0229, 73.3119),
    "Mount Abu": (24.5926, 72.7156),
    "Pushkar": (26.4899, 74.5521),
    "Bharatpur": (27.2176, 77.4895),
    "Kota": (25.2138, 75.8648),
    "Chittorgarh": (24.8887, 74.6269),
    "Alwar": (27.5665, 76.6250),
    "Ranthambore": (26.0173, 76.5026),
    "Sariska": (27.3309, 76.4154),
    "Mandawa": (28.0524, 75.1416),
    "Dungarpur": (23.8430, 73.7142),
    "Bundi": (25.4305, 75.6499),
    "Sikar": (27.6094, 75.1399),
    "Nagaur": (27.2020, 73.7336),
    "Shekhawati": (27.6485, 75.5455),
}


def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


N = len(locations)
cities = list(locations.keys())
D = np.zeros((N, N))

for i in range(N):
    for j in range(i + 1, N):
        D[i, j] = euclidean_distance(locations[cities[i]], locations[cities[j]])
        D[j, i] = D[i, j]


def path_cost_tour(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    cost += distance_matrix[tour[-1], tour[0]]
    return cost


def simulated_annealing(distance_matrix, max_iter=100000, temp_start=1000, cooling_schedule="linear"):
    N = len(distance_matrix)
    current_tour = random.sample(range(N), N)  # Random initial tour
    current_cost = path_cost_tour(current_tour, distance_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost

    cost_history = [current_cost]

    for iteration in range(1, max_iter + 1):
        # Create new tour using 2-opt swap
        i, j = sorted(random.sample(range(N), 2))
        new_tour = (
            current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
        )

        new_cost = path_cost_tour(new_tour, distance_matrix)
        delta_cost = new_cost - current_cost
        
        # Different cooling schedules
        if cooling_schedule == "linear":
            temperature = temp_start * (1 - iteration / max_iter)
        elif cooling_schedule == "exponential":
            temperature = temp_start * (0.95 ** iteration)
        else:  # geometric (default)
            temperature = temp_start / iteration
            
        acceptance_prob = np.exp(-delta_cost / temperature) if delta_cost > 0 and temperature > 0 else 1

        if delta_cost < 0 or random.random() < acceptance_prob:
            current_tour = new_tour
            current_cost = new_cost

        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost

        cost_history.append(best_cost)

    return best_tour, best_cost, cost_history


def create_vlsi_instance(instance_name, num_cities):
    """Create a simple VLSI-like TSP instance"""
    np.random.seed(42)  # For reproducible results
    locations = {}
    for i in range(num_cities):
        locations[f"Node_{i}"] = (np.random.uniform(0, 100), np.random.uniform(0, 100))
    return locations


def run_experiments():
    """Run experiments with different cooling schedules"""
    print("\n" + "="*60)
    print("SIMULATED ANNEALING TSP SOLVER")
    print("="*60)
    
    # Test different cooling schedules
    schedules = ["geometric", "linear", "exponential"]
    results = {}
    
    print("\nTesting different cooling schedules on Rajasthan tour:")
    print("-" * 50)
    
    for schedule in schedules:
        print(f"Running with {schedule} cooling...")
        best_tour, best_cost, cost_history = simulated_annealing(D, cooling_schedule=schedule)
        results[schedule] = {
            'tour': best_tour,
            'cost': best_cost,
            'history': cost_history
        }
        print(f"{schedule:12} cooling: {best_cost:.2f}")
    
    # Find best result
    best_schedule = min(results.keys(), key=lambda x: results[x]['cost'])
    best_result = results[best_schedule]
    
    print(f"\nBest result: {best_schedule} cooling with cost {best_result['cost']:.2f}")
    print(f"Tour: {' -> '.join([cities[i] for i in best_result['tour']])}")
    
    return best_result, results


def solve_vlsi_problems():
    """Solve some VLSI-like problems"""
    print("\n" + "="*60)
    print("VLSI TSP INSTANCES")
    print("="*60)
    
    vlsi_problems = [
        ("Small_VLSI", 50),
        ("Medium_VLSI", 100),
        ("Large_VLSI", 200)
    ]
    
    for name, size in vlsi_problems:
        print(f"\nSolving {name} with {size} nodes...")
        vlsi_locations = create_vlsi_instance(name, size)
        vlsi_cities = list(vlsi_locations.keys())
        
        # Create distance matrix
        vlsi_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                dist = euclidean_distance(vlsi_locations[vlsi_cities[i]], vlsi_locations[vlsi_cities[j]])
                vlsi_matrix[i, j] = vlsi_matrix[j, i] = dist
        
        # Solve with reduced iterations for larger problems
        max_iterations = max(10000, 100000 // (size // 20))
        best_tour, best_cost, cost_history = simulated_annealing(vlsi_matrix, max_iter=max_iterations)
        
        print(f"{name}: Best cost = {best_cost:.2f}, Iterations = {max_iterations}")


# Run the main analysis
best_result, all_results = run_experiments()

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Best tour
plt.subplot(2, 3, 1)
tour_coords = np.array(
    [locations[cities[i]] for i in best_result['tour']] + [locations[cities[best_result['tour'][0]]]]
)
plt.plot(tour_coords[:, 1], tour_coords[:, 0], "o-", linewidth=2, markersize=6, label="Best Tour")
plt.title(f"Best Rajasthan Tour\nCost: {best_result['cost']:.2f}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True, alpha=0.3)

# Add city labels (only for first few to avoid clutter)
for i, city_idx in enumerate(best_result['tour'][:5]):
    plt.text(tour_coords[i, 1], tour_coords[i, 0], cities[city_idx], fontsize=8)

# Plot 2: Cost comparison
plt.subplot(2, 3, 2)
schedule_names = list(all_results.keys())
costs = [all_results[s]['cost'] for s in schedule_names]
plt.bar(schedule_names, costs, alpha=0.7)
plt.title("Cooling Schedule Comparison")
plt.ylabel("Tour Cost")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 3: Convergence comparison
plt.subplot(2, 3, 3)
for schedule in schedule_names:
    history = all_results[schedule]['history']
    # Plot every 1000th point to avoid overcrowding
    indices = range(0, len(history), 1000)
    plt.plot([history[i] for i in indices], label=schedule, alpha=0.8)
plt.title("Convergence Comparison")
plt.xlabel("Iterations (x1000)")
plt.ylabel("Best Cost")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Distance matrix heatmap
plt.subplot(2, 3, 4)
plt.imshow(D, cmap='viridis', aspect='auto')
plt.title("Distance Matrix")
plt.colorbar(label="Distance")

# Plot 5: Tour order
plt.subplot(2, 3, 5)
tour_order = [cities[i] for i in best_result['tour']]
y_pos = np.arange(len(tour_order))
plt.barh(y_pos, range(len(tour_order)))
plt.yticks(y_pos, tour_order)
plt.title("Tour Order")
plt.xlabel("Visit Order")
plt.gca().invert_yaxis()

# Plot 6: Statistics
plt.subplot(2, 3, 6)
plt.axis('off')
stats_text = f"""
Rajasthan Tourism TSP Results

Number of cities: {N}
Best tour cost: {best_result['cost']:.2f}
Average distance per city: {best_result['cost']/N:.2f}

Algorithm: Simulated Annealing
Best cooling: {min(all_results.keys(), key=lambda x: all_results[x]['cost'])}

Cities visited:
{', '.join(cities[:10])}
{'...' if len(cities) > 10 else ''}
"""
plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

# Solve VLSI problems
solve_vlsi_problems()

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print("="*60)
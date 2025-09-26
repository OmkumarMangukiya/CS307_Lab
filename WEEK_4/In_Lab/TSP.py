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


def simulated_annealing(distance_matrix, max_iter=100000, temp_start=1000):
    N = len(distance_matrix)
    current_tour = random.sample(range(N), N)
    current_cost = path_cost_tour(current_tour, distance_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost

    for iteration in range(1, max_iter + 1):
        # Create new tour using 2-opt swap
        i, j = sorted(random.sample(range(N), 2))
        new_tour = (
            current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
        )

        new_cost = path_cost_tour(new_tour, distance_matrix)
        delta_cost = new_cost - current_cost
        
        temperature = temp_start * (1 - iteration / max_iter)
        acceptance_prob = np.exp(-delta_cost / temperature) if delta_cost > 0 and temperature > 0 else 1

        if delta_cost < 0 or random.random() < acceptance_prob:
            current_tour = new_tour
            current_cost = new_cost

        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost

    return best_tour, best_cost


# Solve TSP using Simulated Annealing
print("Solving TSP for Rajasthan cities...")
best_tour, best_cost = simulated_annealing(D)

print(f"Best tour cost: {best_cost:.2f}")
print(f"Tour: {' -> '.join([cities[i] for i in best_tour])}")

# Visualize the best tour
plt.figure(figsize=(10, 8))
tour_coords = np.array(
    [locations[cities[i]] for i in best_tour] + [locations[cities[best_tour[0]]]]
)

plt.plot(tour_coords[:, 1], tour_coords[:, 0], "o-", linewidth=2, markersize=8, 
         color='red', label=f"Best Tour (Cost: {best_cost:.2f})")

# Add city labels
for i, city_idx in enumerate(best_tour):
    city_name = cities[city_idx]
    plt.text(tour_coords[i, 1], tour_coords[i, 0], city_name, fontsize=9, 
             ha='center', va='bottom')

plt.title("Rajasthan Tourism TSP Solution", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
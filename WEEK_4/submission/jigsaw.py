import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
from tqdm import tqdm

from collections import deque
import time
import psutil


def load_octave_column_matrix(file_path):
    """
    Load an octave-style column matrix stored in a text-like file where the first 5 lines are header
    and the following lines contain one integer per line (512*512 lines expected).
    Returns a (512, 512) numpy array.
    """
    matrix = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    matrix_lines = lines[5:]

    for line in matrix_lines:
        line = line.strip()
        if line:
            try:
                matrix.append(int(line))
            except ValueError:
                print(f"Skipping invalid line: {line}")

    matrix = np.array(matrix, dtype=np.int32)

    if matrix.size != 512 * 512:
        raise ValueError(f"Expected 262144 elements, but got {matrix.size} elements.")

    reshaped_matrix = matrix.reshape((512, 512))
    return reshaped_matrix


def edge_difference(edge1, edge2):
    diff = np.abs(edge1.astype(float) - edge2.astype(float))
    return np.mean(diff)

def calculate_edge_compatibility(piece1, piece2, direction):
    """Calculate compatibility score between two pieces in given direction"""
    if direction == 'up':
        edge1 = piece1[0, :]
        edge2 = piece2[-1, :]
    elif direction == 'down':
        edge1 = piece1[-1, :]
        edge2 = piece2[0, :]
    elif direction == 'left':
        edge1 = piece1[:, 0]
        edge2 = piece2[:, -1]
    elif direction == 'right':
        edge1 = piece1[:, -1]
        edge2 = piece2[:, 0]
    
    base_diff = edge_difference(edge1, edge2)
    
    corner_bonus = 0
    if direction in ['up', 'down']:
        corner_diff1 = abs(float(edge1[0]) - float(edge2[0]))
        corner_diff2 = abs(float(edge1[-1]) - float(edge2[-1]))
        corner_bonus = (corner_diff1 + corner_diff2) * 0.5
    else:
        corner_diff1 = abs(float(edge1[0]) - float(edge2[0]))
        corner_diff2 = abs(float(edge1[-1]) - float(edge2[-1]))
        corner_bonus = (corner_diff1 + corner_diff2) * 0.5
    
    return base_diff + corner_bonus * 0.3


def get_value(grid, pieces):
    total_difference = 0
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            piece_index = grid[i][j]
            if piece_index is not None:
                piece = pieces[piece_index]
                if i > 0 and grid[i-1][j] is not None:
                    total_difference += calculate_edge_compatibility(piece, pieces[grid[i-1][j]], 'up')
                if i < rows-1 and grid[i+1][j] is not None:
                    total_difference += calculate_edge_compatibility(piece, pieces[grid[i+1][j]], 'down')
                if j > 0 and grid[i][j-1] is not None:
                    total_difference += calculate_edge_compatibility(piece, pieces[grid[i][j-1]], 'left')
                if j < cols-1 and grid[i][j+1] is not None:
                    total_difference += calculate_edge_compatibility(piece, pieces[grid[i][j+1]], 'right')
    return total_difference


def generate_neighbor(grid):
    new_grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    
    if random.random() < 0.8:
        i1, j1 = random.randint(0, rows-1), random.randint(0, cols-1)
        
        nearby_positions = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i1 + di, j1 + dj
                if 0 <= ni < rows and 0 <= nj < cols and (di != 0 or dj != 0):
                    nearby_positions.append((ni, nj))
        
        if nearby_positions:
            i2, j2 = random.choice(nearby_positions)
        else:
            i2, j2 = random.randint(0, rows-1), random.randint(0, cols-1)
    else:
        i1, j1 = random.randint(0, rows-1), random.randint(0, cols-1)
        i2, j2 = random.randint(0, rows-1), random.randint(0, cols-1)
    
    new_grid[i1][j1], new_grid[i2][j2] = new_grid[i2][j2], new_grid[i1][j1]
    return new_grid


def greedy_initial_placement(pieces):
    """Create a better initial arrangement using greedy edge matching"""
    rows, cols = 4, 4
    grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    used_pieces = set()
    
    corner_piece = random.randint(0, len(pieces) - 1)
    grid[0][0] = corner_piece
    used_pieces.add(corner_piece)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != -1:
                continue
                
            best_piece = -1
            best_score = float('inf')
            
            for piece_idx in range(len(pieces)):
                if piece_idx in used_pieces:
                    continue
                    
                total_score = 0
                valid_neighbors = 0
                
                if i > 0 and grid[i-1][j] != -1:
                    score = calculate_edge_compatibility(pieces[piece_idx], pieces[grid[i-1][j]], 'up')
                    total_score += score
                    valid_neighbors += 1
                    
                if j > 0 and grid[i][j-1] != -1:
                    score = calculate_edge_compatibility(pieces[piece_idx], pieces[grid[i][j-1]], 'left')
                    total_score += score
                    valid_neighbors += 1
                
                if valid_neighbors > 0:
                    avg_score = total_score / valid_neighbors
                else:
                    avg_score = 0
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_piece = piece_idx
            
            if best_piece != -1:
                grid[i][j] = best_piece
                used_pieces.add(best_piece)
    
    return grid

def simulated_annealing(pieces, initial_grid, max_iterations=20000, initial_temp=2000, cooling_rate=0.9995):
    current_grid = copy.deepcopy(initial_grid)
    current_value = get_value(current_grid, pieces)
    best_grid = copy.deepcopy(current_grid)
    best_value = current_value
    temperature = initial_temp
    
    improvements = 0
    last_improvement = 0

    for iteration in tqdm(range(max_iterations), desc="Solving jigsaw"):
        neighbor_grid = generate_neighbor(current_grid)
        neighbor_value = get_value(neighbor_grid, pieces)
        delta = neighbor_value - current_value

        acceptance_prob = math.exp(-delta / max(temperature, 1e-10)) if delta > 0 else 1.0
        
        if delta < 0 or random.uniform(0, 1) < acceptance_prob:
            current_grid = neighbor_grid
            current_value = neighbor_value

            if current_value < best_value:
                best_grid = copy.deepcopy(current_grid)
                best_value = current_value
                improvements += 1
                last_improvement = iteration

        temperature *= cooling_rate
        if iteration - last_improvement > max_iterations // 4:
            print(f"\nEarly stopping at iteration {iteration} (no improvement for {iteration - last_improvement} iterations)")
            break

    print(f"\nOptimization completed with {improvements} improvements. Final score: {best_value:.2f}")
    return best_grid, best_value


def multiple_runs_optimization(pieces, num_runs=3):
    """Run multiple optimization attempts with different initial configurations"""
    best_overall_grid = None
    best_overall_value = float('inf')
    
    print(f"Running {num_runs} optimization attempts...")
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        if run == 0:
            print("Using greedy initial placement...")
            initial_grid = greedy_initial_placement(pieces)
        elif run == 1:
            print("Using sequential initial placement...")
            initial_grid = [[i*4 + j for j in range(4)] for i in range(4)]
        else:
            print("Using random initial placement...")
            pieces_indices = list(range(16))
            random.shuffle(pieces_indices)
            initial_grid = []
            idx = 0
            for i in range(4):
                row = []
                for j in range(4):
                    row.append(pieces_indices[idx])
                    idx += 1
                initial_grid.append(row)
        
        initial_score = get_value(initial_grid, pieces)
        print(f"Initial score: {initial_score:.2f}")
        
        grid, value = simulated_annealing(pieces, initial_grid)
        improvement = initial_score - value
        print(f"Improvement: {improvement:.2f} ({(improvement/initial_score)*100:.1f}%)")
        
        if value < best_overall_value:
            best_overall_grid = copy.deepcopy(grid)
            best_overall_value = value
            print(f"New best result found in run {run + 1}!")
    
    print(f"\nBest overall score: {best_overall_value:.2f}")
    return best_overall_grid, best_overall_value


def reconstruct_image(pieces, grid):
    piece_height, piece_width = pieces[0].shape
    rows, cols = len(grid), len(grid[0])
    reconstructed = np.zeros((rows * piece_height, cols * piece_width))

    for i in range(rows):
        for j in range(cols):
            piece_index = grid[i][j]
            reconstructed[i*piece_height:(i+1)*piece_height, j*piece_width:(j+1)*piece_width] = pieces[piece_index]

    return reconstructed


print("Loading jigsaw puzzle...")
pieces = []
matrix = load_octave_column_matrix("jigsaw.mat")

piece_size = 128
num_pieces_per_side = 4
rows, cols = 4, 4

for i in range(num_pieces_per_side):
    for j in range(num_pieces_per_side):
        piece = matrix[i*piece_size:(i+1)*piece_size, j*piece_size:(j+1)*piece_size]
        pieces.append(piece)

print(f"Created {len(pieces)} puzzle pieces of size {piece_size}x{piece_size}")

random.seed(42)
np.random.seed(42)

start_time = time.time()
final_grid, final_value = multiple_runs_optimization(pieces, num_runs=3)
total_time = time.time() - start_time

print(f"\nOptimization completed in {total_time:.2f} seconds")

original_grid = [[i*cols + j for j in range(cols)] for i in range(rows)]
original_image = reconstruct_image(pieces, original_grid)
result_image = reconstruct_image(pieces, final_grid)

original_score = get_value(original_grid, pieces)
improvement = original_score - final_value
improvement_percentage = (improvement / original_score) * 100

print(f"\nResults:")
print(f"Original score: {original_score:.2f}")
print(f"Final score: {final_value:.2f}")
print(f"Improvement: {improvement:.2f} ({improvement_percentage:.1f}%)")

plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.title(f"Original Scrambled\n(Score: {original_score:.1f})", fontsize=12)
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Reconstructed\n(Score: {final_value:.1f})", fontsize=12)
plt.imshow(result_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
difference = np.abs(original_image.astype(float) - result_image.astype(float))
plt.title(f"Difference\n(Improvement: {improvement_percentage:.1f}%)", fontsize=12)
plt.imshow(difference, cmap='hot')
plt.colorbar(shrink=0.8)
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\nPuzzle solving complete! Check the visualization window for results.")
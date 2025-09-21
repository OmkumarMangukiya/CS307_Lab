from collections import deque

def display_configuration(config):
    """Display the current configuration of the puzzle."""
    return ' '.join('E' if x == 1 else 'W' if x == 0 else '_' for x in config)

def is_solved(config):
    """Check if the puzzle is solved."""
    return config == [0, 0, 0, 2, 1, 1, 1]

def generate_moves(config):
    """Generate all possible moves from the current configuration."""
    possible_moves = []
    for i, rabbit in enumerate(config):
        if rabbit == 1:  
            if i + 1 < len(config) and config[i + 1] == 2:
                possible_moves.append((i, i + 1))
            if i + 2 < len(config) and config[i + 1] == 0 and config[i + 2] == 2:
                possible_moves.append((i, i + 2))
        elif rabbit == 0:
            if i - 1 >= 0 and config[i - 1] == 2:
                possible_moves.append((i, i - 1))
            if i - 2 >= 0 and config[i - 1] == 1 and config[i - 2] == 2:
                possible_moves.append((i, i - 2))
    return possible_moves

def apply_move(config, move):
    """Apply a move to the configuration."""
    new_config = config[:]
    new_config[move[1]] = new_config[move[0]]
    new_config[move[0]] = 2
    return new_config

def solve_puzzle(start_config):
    """Solve the Rabbit Hop puzzle using breadth-first search."""
    queue = deque([(start_config, [])])
    seen = set()

    while queue:
        current_config, history = queue.popleft()

        if tuple(current_config) in seen:
            continue

        seen.add(tuple(current_config))
        history = history + [current_config]

        if is_solved(current_config):
            return history

        for move in generate_moves(current_config):
            next_config = apply_move(current_config, move)
            queue.append((next_config, history))

    return None

start_config = [1, 1, 1, 2, 0, 0, 0]
solution = solve_puzzle(start_config)

if solution:
    print("Solution found:")
    for step, config in enumerate(solution):
        print(f"Step {step}: {display_configuration(config)}")
    print(f"Total moves: {len(solution) - 1}")
else:
    print("No solution exists.")
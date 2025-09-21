def visualize_board(board):
    """Visualize the current board state."""
    symbols = {1: '>', 0: '<', -1: '_'}
    return ' '.join(symbols[x] for x in board)

def is_final_configuration(board):
    """Check if the board is in its final configuration."""
    return board == [0, 0, 0, -1, 1, 1, 1]

def find_valid_jumps(board):
    """Determine all valid jumps for the current board state."""
    valid_jumps = []
    for i, rabbit in enumerate(board):
        if rabbit == 1:  
            if i + 1 < len(board) and board[i + 1] == -1:
                valid_jumps.append((i, i + 1))
            if i + 2 < len(board) and board[i + 1] == 0 and board[i + 2] == -1:
                valid_jumps.append((i, i + 2))
        elif rabbit == 0:
            if i - 1 >= 0 and board[i - 1] == -1:
                valid_jumps.append((i, i - 1))
            if i - 2 >= 0 and board[i - 1] == 1 and board[i - 2] == -1:
                valid_jumps.append((i, i - 2))
    return valid_jumps

def execute_jump(board, jump):
    """Execute a jump on the board."""
    new_board = board.copy()
    new_board[jump[1]] = new_board[jump[0]]
    new_board[jump[0]] = -1
    return new_board

def solve_puzzle(initial_board):
    """Solve the Rabbit Jump puzzle using depth-first search."""
    stack = [(initial_board, [])]
    explored = set()

    while stack:
        current_board, move_history = stack.pop()

        if tuple(current_board) in explored:
            continue

        explored.add(tuple(current_board))
        move_history = move_history + [current_board]

        if is_final_configuration(current_board):
            return move_history

        for jump in find_valid_jumps(current_board):
            next_board = execute_jump(current_board, jump)
            stack.append((next_board, move_history))

    return None

# Starting configuration: > > > _ < < <
start_board = [1, 1, 1, -1, 0, 0, 0]
solution = solve_puzzle(start_board)

if solution:
    print("Solution discovered:")
    for step, board in enumerate(solution):
        print(f"Move {step}: {visualize_board(board)}")
    print(f"Total jumps required: {len(solution) - 1}")
else:
    print("No solution exists for this puzzle.")
import heapq

class SolitaireNode:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = w1 * g + w2 * h

    def __lt__(self, other):
        return self.f < other.f

def get_possible_moves(state):
    moves = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    jump_over = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                for (dr, dc), (jr, jc) in zip(directions, jump_over):
                    end_r = r + dr
                    end_c = c + dc
                    jump_r = r + jr
                    jump_c = c + jc
                    if 0 <= end_r < 7 and 0 <= end_c < 7:
                        if state[jump_r][jump_c] == 'O' and state[end_r][end_c] == '0':
                            moves.append((r, c, end_r, end_c))
    return moves

def apply_move(state, move):
    new_state = [row[:] for row in state]
    start_row, start_col, end_row, end_col = move
    jump_row = (start_row + end_row) // 2
    jump_col = (start_col + end_col) // 2
    new_state[end_row][end_col] = 'O'
    new_state[start_row][start_col] = '0'
    new_state[jump_row][jump_col] = '0'
    return new_state

def heuristic_1(state):
    return sum(row.count('O') for row in state)

def heuristic_2(state):
    center = (3, 3)
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                total_distance += abs(r - center[0]) + abs(c - center[1])
    return total_distance

def best_first_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.h, start_node))
    visited = set()
    max_size=0
    while open_list:
        _, node = heapq.heappop(open_list)
        if len(open_list)>max_size:
            max_size=len(open_list)
        if tuple(map(tuple, node.state)) in visited:
            continue
        visited.add(tuple(map(tuple, node.state)))
        if heuristic_func(node.state) == 0:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("max number of nodes in queue is: ", max_size)
            print("number of nodes visited is: ", len(visited))
            print("number of nodes in the path is: ", len(path))
            return path[::-1]
        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, node, h=h)
            heapq.heappush(open_list, (new_node.h, new_node))
    return None

def a_star_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    max_size=0
    while open_list:
        _, node = heapq.heappop(open_list)
        if len(open_list)>max_size:
            max_size=len(open_list)
        if tuple(map(tuple, node.state)) in visited:
            continue
        visited.add(tuple(map(tuple, node.state)))
        if heuristic_func(node.state) == 0:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("max number of nodes in queue is: ", max_size)
            print("number of nodes visited is: ", len(visited))
            print("number of nodes in the path is: ", len(path))
            return path[::-1]
        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            g = node.g + 1
            h = heuristic_func(new_state)
            f = g + h
            new_node = SolitaireNode(new_state, node, g=g, h=h)
            heapq.heappush(open_list, (new_node.f, new_node))
    return None

start_state = [
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', '0', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['-', '-', 'O', 'O', 'O', '-', '-']
]

moves = get_possible_moves(start_state)
for move in moves:
    print(f"Start: {move[0], move[1]} -> End: {move[2], move[3]}")
print("Best first search results are: ")
best_first_search(start_state, heuristic_2)
print("A* search results are: ")
a_star_search(start_state, heuristic_2)

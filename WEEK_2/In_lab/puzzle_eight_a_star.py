import heapq
import random

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other):    
        return self.f < other.f

def heuristic(node, goal_state):
    # Example heuristic: number of misplaced tiles
    h = sum(1 for i, val in enumerate(node.state) if val != 0 and val != goal_state[i])
    return h
def get_successors(node):
    successors = []
    index = node.state.index(0)
    quotient = index//3
    remainder = index%3
    if quotient == 0:
        moves = [3]
    if quotient == 1:
        moves = [-3, 3]
    if quotient == 2:
        moves = [-3]
    if remainder == 0:
        moves += [1]
    if remainder == 1:
        moves += [-1, 1]
    if remainder == 2:
        moves += [-1]
    for move in moves:
        im = index+move
        if im >= 0 and im < 9:
            new_state = list(node.state)
            temp = new_state[im]
            new_state[im] = new_state[index]
            new_state[index] = temp
            successor = Node(new_state, node, node.g+1)
            successors.append(successor)            
    return successors

def search_agent(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    frontier = []
    heapq.heappush(frontier, (start_node.g, start_node))
    visited = set()
    nodes_explored = 0
    while frontier:
        _, node = heapq.heappop(frontier)
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]
        for successor in get_successors(node):
            heapq.heappush(frontier, (successor.g, successor))
    return None

start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
s_node = Node(start_state)
D = 20
d = 0
while d <= D:
    goal_state = random.choice(list(get_successors(s_node))).state
    s_node = Node(goal_state)
    d = d+1

solution = search_agent(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")

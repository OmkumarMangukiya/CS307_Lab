from generator import generate_k_sat_problem
import random

class Node:
    def __init__(self, state):
        self.state = state

def heuristic_value_1(clause, node):
    count = 0
    for curr_clause in clause:
        for i in curr_clause:
            if i > 0 and node.state[i - 1] == 1:
                count += 1
                break
            if i < 0 and node.state[abs(i) - 1] == 0:
                count += 1
                break
    return count

def heuristic_value_2(clause, node):
    state = node.state
    count = 0
    for curr_clause in clause:
        for literal in curr_clause:
            if state[abs(literal) - 1] == 1:
                count += 1
    return count

def check(clause, node):
    count = 0
    for curr_clause in clause:
        for i in curr_clause:
            if i > 0 and node.state[i - 1] == 1:
                count += 1
                break
            if i < 0 and node.state[abs(i) - 1] == 0:
                count += 1
                break
    if count == len(clause):
        return True
    return False

def gen_sucessors(node, clause):
    max = -1
    max_node = node
    i = random.randint(0, len(node.state) - 1)
    for i in range(len(node.state)):
        temp = node.state.copy()
        if temp[i] == 0:
            temp[i] = 1
        elif temp[i] == 1:
            temp[i] = 0
        new_node = Node(state=temp)
        val = heuristic_value_2(clause, new_node)
        if val > max:
            max = val
            max_node = new_node
    if max_node.state == node.state:
        return None
    return max_node

def calculate_penetrance(num_instances, k, m, n):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = hill_climb(clauses, k, m, n)
        if is_solved:
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100
    return penetrance

def hill_climb(clause, k, m, n, max_iter=1000):
    node = Node([0] * n)
    for i in range(max_iter):
        if check(clause, node):
            print(f"clause is {clause}")
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Steps required to reach solution {i}")
            return True
        node = gen_sucessors(node, clause)
        if node is None:
            print("Local minima reached")
            return False
    return False

clause = generate_k_sat_problem(3, 100, 100)
print(clause)
print(calculate_penetrance(20, 3, 50, 50))

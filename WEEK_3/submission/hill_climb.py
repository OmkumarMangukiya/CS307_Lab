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

def gen_sucessors(node, clause, use_h1=True):
    max_val = -1
    max_node = node
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]  # Flip bit
        new_node = Node(state=temp)
        val = heuristic_value_1(clause, new_node) if use_h1 else heuristic_value_2(clause, new_node)
        if val > max_val:
            max_val = val
            max_node = new_node
    if max_node.state == node.state:
        return None
    return max_node

def calculate_penetrance(num_instances, k, m, n, use_h1=True):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = hill_climb(clauses, k, m, n, use_h1=use_h1)
        if is_solved:
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100
    return penetrance

def hill_climb(clause, k, m, n, max_iter=1000, use_h1=True):
    node = Node([random.choice([0, 1]) for _ in range(n)])
    for i in range(max_iter):
        if check(clause, node):
            if i < max_iter - 1:
                print(f"clause is {clause}")
                print("Solution found")
                print(f"Solution is{node.state}")
                print(f"Steps required to reach solution {i}")
            return True
        node = gen_sucessors(node, clause, use_h1)
        if node is None:
            if i < max_iter - 1:
                print("Local minima reached")
            return False
    return False


hc_results = {}
hc_results['H1'] = [
    calculate_penetrance(20, 3, 10, 10, use_h1=True),
    calculate_penetrance(20, 3, 25, 25, use_h1=True),
    calculate_penetrance(20, 3, 50, 50, use_h1=True)
]
hc_results['H2'] = [
    calculate_penetrance(20, 3, 10, 10, use_h1=False),
    calculate_penetrance(20, 3, 25, 25, use_h1=False),
    calculate_penetrance(20, 3, 50, 50, use_h1=False)
]

print("\nHill Climb Final Penetrance Results:")
print("H1: (10,10):", hc_results['H1'][0], ", (25,25):", hc_results['H1'][1], ", (50,50):", hc_results['H1'][2])
print("H2: (10,10):", hc_results['H2'][0], ", (25,25):", hc_results['H2'][1], ", (50,50):", hc_results['H2'][2])

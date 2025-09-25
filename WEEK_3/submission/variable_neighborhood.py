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
    if node is None:
        return False
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

def gen_1(node, clause, use_h1=True):
    max_val = -1
    max_node = node
    count = 0
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        new_node = Node(state=temp)
        val = heuristic_value_1(clause, new_node) if use_h1 else heuristic_value_2(clause, new_node)
        if val > max_val:
            max_val = val
            max_node = new_node
        else:
            count += 1
    if count == len(node.state):
        return None
    return max_node

def gen_2(node, clause, num_neighbors=20, use_h1=True):
    max_value = heuristic_value_1(clause, node) if use_h1 else heuristic_value_2(clause, node)
    max_node = node
    improved = False
    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_bits_to_flip = random.choice([1, 2])
        if num_bits_to_flip == 1:
            i = random.randint(0, len(node.state) - 1)
            temp[i] = 1 - temp[i]
        elif num_bits_to_flip == 2:
            i, j = random.sample(range(len(node.state)), 2)
            temp[i] = 1 - temp[i]
            temp[j] = 1 - temp[j]
        new_node = Node(state=temp)
        val = heuristic_value_1(clause, new_node) if use_h1 else heuristic_value_2(clause, new_node)
        if val > max_value:
            max_value = val
            max_node = new_node
            improved = True
    if not improved:
        return None
    return max_node

def gen_3(node, clause, num_neighbors=30, use_h1=True):
    max_value = heuristic_value_1(clause, node) if use_h1 else heuristic_value_2(clause, node)
    max_node = node
    improved = False
    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_bits_to_flip = random.choice([1, 2, 3])
        positions = random.sample(range(len(node.state)), num_bits_to_flip)
        for pos in positions:
            temp[pos] = 1 - temp[pos]
        new_node = Node(state=temp)
        val = heuristic_value_1(clause, new_node) if use_h1 else heuristic_value_2(clause, new_node)
        if val > max_value:
            max_value = val
            max_node = new_node
            improved = True
        elif val == max_value and random.random() < 0.1:  
            max_node = new_node
            improved = True
    if not improved:
        return None
    return max_node

def calculate_penetrance(num_instances, k, m, n, use_h1=True):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = vgn(clauses, k, m, n, use_h1=use_h1)
        if is_solved:
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100
    return penetrance

def hill_climb(clause, node, gen_func, k, m, n, max_iter=1000, use_h1=True):
    prev_node = node
    for i in range(max_iter):
        if check(clause, node):
            print(f"clause is {clause}")
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Steps required to reach solution {i}")
            return node
        if node is None:
            print("Local minima reached")
            print(prev_node.state)
            return prev_node
        temp_node = gen_func(node, clause, use_h1=use_h1)
        prev_node = node
        node = temp_node
    return node

def vgn(clause, k, m, n, max_restarts=3, use_h1=True):
    best_node = None
    best_value = -1
    for _ in range(max_restarts):
        node = Node([random.choice([0, 1]) for _ in range(n)])
        node = hill_climb(clause, node, gen_1, k, m, n, use_h1=use_h1)
        if check(clause, node):
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Node reached after gen_1")
            return True
        print("GEEn 2")
        node = hill_climb(clause, node, gen_2, k, m, n, use_h1=use_h1)
        if check(clause, node):
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Node reached after gen_2")
            return True
        print("GEEn 3")
        node = hill_climb(clause, node, gen_3, k, m, n, use_h1=use_h1)
        if check(clause, node):
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Node reached after gen_3")
            return True
        current_value = heuristic_value_1(clause, node) if use_h1 else heuristic_value_2(clause, node)
        if current_value > best_value:
            best_value = current_value
            best_node = node
    return check(clause, best_node)



vns_results = {}
vns_results['H1'] = [
    calculate_penetrance(20, 3, 10, 10, use_h1=True),
    calculate_penetrance(20, 3, 25, 25, use_h1=True),
    calculate_penetrance(20, 3, 50, 50, use_h1=True)
]
vns_results['H2'] = [
    calculate_penetrance(20, 3, 10, 10, use_h1=False),
    calculate_penetrance(20, 3, 25, 25, use_h1=False),
    calculate_penetrance(20, 3, 50, 50, use_h1=False)
]

print("\nVNS Final Penetrance Results:")
print("H1: (10,10):", vns_results['H1'][0], ", (25,25):", vns_results['H1'][1], ", (50,50):", vns_results['H1'][2])
print("H2: (10,10):", vns_results['H2'][0], ", (25,25):", vns_results['H2'][1], ", (50,50):", vns_results['H2'][2])

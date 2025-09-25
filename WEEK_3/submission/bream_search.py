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
        val = heuristic_value_1(clause, new_node)
        if val > max:
            max = val
            max_node = new_node
    if max_node.state == node.state:
        return None
    return max_node

def generate_successors(node, clause, beam_width=3, use_h1=True):
    successors = []
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        new_node = Node(state=temp)
        successors.append(new_node)
    heuristic = heuristic_value_1 if use_h1 else heuristic_value_2
    successors.sort(key=lambda x: heuristic(clause, x), reverse=True)
    return successors[:beam_width]

def calculate_penetrance(num_instances, k, m, n, use_h1=True):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = beam(clauses, k, m, n, use_h1=use_h1)
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
            return node
        node = gen_sucessors(node, clause)
        if node is None:
            print("Local minima reached")
            return None

def beam(clause, k, m, n, max_iter=1000, beam_width=3, use_h1=True):
    node = Node([random.choice([0, 1]) for _ in range(n)])
    if check(clause, node):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Steps required to reach solution are: 0")
        return True
    count = 0
    sucessors = generate_successors(node, clause, beam_width, use_h1=use_h1)
    for i in range(max_iter):
        new_sucessors = []
        if sucessors == []:
            return False
        for sucessor in sucessors:
            if check(clause, sucessor):
                count += 1
                print(count)
                print("Solution found")
                print(f"Solution is{sucessor.state}")
                print(f"Steps required to reach solution are: {i+1}")
                return True
            temp = gen_sucessors(sucessor, clause)
            new_sucessors.append(temp)
        sucessors = new_sucessors

beam_results = {}
beam_results['H1'] = [
    calculate_penetrance(20, 3, 5, 5, use_h1=True),
    calculate_penetrance(20, 3, 10, 10, use_h1=True),
    calculate_penetrance(20, 3, 25, 25, use_h1=True)
]
beam_results['H2'] = [
    calculate_penetrance(20, 3, 5, 5, use_h1=False),
    calculate_penetrance(20, 3, 10, 10, use_h1=False),
    calculate_penetrance(20, 3, 25, 25, use_h1=False)
]

print("\nBeam Search Final Penetrance Results:")
print("H1: (5,5):", beam_results['H1'][0], ", (10,10):", beam_results['H1'][1], ", (25,25):", beam_results['H1'][2])
print("H2: (5,5):", beam_results['H2'][0], ", (10,10):", beam_results['H2'][1], ", (25,25):", beam_results['H2'][2])

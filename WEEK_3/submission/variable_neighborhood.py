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

def gen_1(node, clause):
    max = -1
    max_node = node
    count = 0
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
        else:
            count += 1
    if count == len(node.state):
        print("yes")
        return None
    return max_node

def gen_2(node, clause, num_neighbors=10):
    max_value = -1
    max_node = node
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
        val = heuristic_value_2(clause, new_node)
        if val > max_value:
            max_value = val
            max_node = new_node
    if max_node.state == node.state:
        return None
    return max_node

def gen_3(node, clause, num_neighbors=10):
    max_value = -1
    max_node = node
    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_bits_to_flip = random.choice([1, 2, 3])
        if num_bits_to_flip == 1:
            i = random.randint(0, len(node.state) - 1)
            temp[i] = 1 - temp[i]
        elif num_bits_to_flip == 2:
            i, j = random.sample(range(len(node.state)), 2)
            temp[i] = 1 - temp[i]
            temp[j] = 1 - temp[j]
        elif num_bits_to_flip == 3:
            i, j, k = random.sample(range(len(node.state)), 3)
            temp[i] = 1 - temp[i]
            temp[j] = 1 - temp[j]
            temp[k] = 1 - temp[k]
        new_node = Node(state=temp)
        val = heuristic_value_2(clause, new_node)
        if val > max_value:
            max_value = val
            max_node = new_node
    if max_node.state == node.state:
        return None
    return max_node

def calculate_penetrance(num_instances, k, m, n):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = vgn(clauses, k, m, n)
        if is_solved:
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100
    return penetrance

def hill_climb(clause, node, gen_func, k, m, n, max_iter=1000):
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
        temp_node = gen_func(node, clause)
        prev_node = node
        node = temp_node
    return node

def vgn(clause, k, m, n):
    node = Node([0] * n)
    node = hill_climb(clause, node, gen_1, k, m, n)
    if check(clause, node):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_1")
        return node
    print("GEEn 2")
    print(node.state)
    node = hill_climb(clause, node, gen_2, k, m, n)
    if check(clause, node):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_2")
        return node
    print("GEEn 3")
    node = hill_climb(clause, node, gen_3, k, m, n)
    if check(clause, node):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_3")
        return node
    if check(clause, node):
        return True
    else:
        return False

clause = generate_k_sat_problem(3, 75, 75)
print(calculate_penetrance(20, 3, 10, 10))

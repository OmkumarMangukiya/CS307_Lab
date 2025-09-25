import random

def generate_k_sat_problem(k, m, n):
    clauses = []

    for _ in range(m):
        clause = set()  
        
        while len(clause) < k:
            var = random.randint(1, n) 
            is_negated = random.choice([True, False])  
            literal = -var if is_negated else var
            clause.add(literal)
        
    
        clauses.append(sorted(clause, key=abs))
    
    return clauses

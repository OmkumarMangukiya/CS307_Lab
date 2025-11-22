import numpy as np
from functools import lru_cache
from scipy.stats import poisson

MAX_BIKES = 20
MAX_MOVE = 5
DISCOUNT = 0.9
# limit Poisson enumeration for speed; last bucket absorbs remaining mass
POISSON_CAP = 11
POISSON_RANGE = POISSON_CAP + 1


def truncated_poisson_probs(lam):
    probs = np.array([poisson.pmf(k, lam) for k in range(POISSON_RANGE)])
    probs[-1] += 1.0 - probs.sum()
    return probs


PR_RENT_1 = truncated_poisson_probs(3)
PR_RENT_2 = truncated_poisson_probs(4)
PR_RET_1 = truncated_poisson_probs(3)
PR_RET_2 = truncated_poisson_probs(2)


def valid_action_bounds(state):
    s1, s2 = state
    min_action = max(-MAX_MOVE, s1 - MAX_BIKES, -s2)
    max_action = min(MAX_MOVE, s1, MAX_BIKES - s2)
    return range(int(min_action), int(max_action) + 1)


@lru_cache(maxsize=None)
def transition_bundle(s1, s2, action):
    bikes1 = min(MAX_BIKES, max(0, s1 - action))
    bikes2 = min(MAX_BIKES, max(0, s2 + action))

    outcomes = {}
    for rent1, p_rent1 in enumerate(PR_RENT_1):
        for rent2, p_rent2 in enumerate(PR_RENT_2):
            rented1 = min(bikes1, rent1)
            rented2 = min(bikes2, rent2)
            remaining1 = bikes1 - rented1
            remaining2 = bikes2 - rented2

            for ret1, p_ret1 in enumerate(PR_RET_1):
                for ret2, p_ret2 in enumerate(PR_RET_2):
                    prob = p_rent1 * p_rent2 * p_ret1 * p_ret2
                    if prob == 0:
                        continue

                    bikes_after1 = min(MAX_BIKES, remaining1 + ret1)
                    bikes_after2 = min(MAX_BIKES, remaining2 + ret2)
                    next_state = (bikes_after1, bikes_after2)
                    reward = 10 * (rented1 + rented2)

                    if next_state not in outcomes:
                        outcomes[next_state] = [0.0, 0.0]
                    outcomes[next_state][0] += prob
                    outcomes[next_state][1] += prob * reward

    packed = tuple((ns, prob, reward_sum) for ns, (prob, reward_sum) in outcomes.items())
    return (bikes1, bikes2), packed


def expected_return(state, action, values, transfer_cost_fn, parking_penalty_fn):
    (pre_b1, pre_b2), transitions = transition_bundle(state[0], state[1], action)
    total = -transfer_cost_fn(action) - parking_penalty_fn(pre_b1, pre_b2)

    for next_state, prob, reward_sum in transitions:
        total += reward_sum + DISCOUNT * prob * values[next_state]
    return total


def policy_evaluation(policy, values, transfer_cost_fn, parking_penalty_fn, theta=1e-3):
    while True:
        delta = 0.0
        old_values = values.copy()
        new_values = old_values.copy()

        for s1 in range(MAX_BIKES + 1):
            for s2 in range(MAX_BIKES + 1):
                action = policy[s1, s2]
                new_val = expected_return((s1, s2), action, old_values, transfer_cost_fn, parking_penalty_fn)
                new_values[s1, s2] = new_val
                delta = max(delta, abs(new_val - old_values[s1, s2]))

        values[:, :] = new_values
        if delta < theta:
            break


def policy_improvement(policy, values, transfer_cost_fn, parking_penalty_fn):
    policy_stable = True
    for s1 in range(MAX_BIKES + 1):
        for s2 in range(MAX_BIKES + 1):
            old_action = policy[s1, s2]
            best_value = float("-inf")
            best_action = old_action

            for action in valid_action_bounds((s1, s2)):
                val = expected_return((s1, s2), action, values, transfer_cost_fn, parking_penalty_fn)
                if val > best_value:
                    best_value = val
                    best_action = action

            policy[s1, s2] = best_action
            if best_action != old_action:
                policy_stable = False

    return policy_stable


def policy_iteration(transfer_cost_fn, parking_penalty_fn):
    values = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros_like(values, dtype=int)
    iteration = 0

    while True:
        iteration += 1
        print(f"Policy iteration pass {iteration}...")
        policy_evaluation(policy, values, transfer_cost_fn, parking_penalty_fn)
        if policy_improvement(policy, values, transfer_cost_fn, parking_penalty_fn):
            break

    return policy, values


def base_transfer_cost(action):
    return 2 * abs(action)


def modified_transfer_cost(action):
    if action > 0:
        return 2 * max(0, action - 1)
    return 2 * abs(action)


def no_parking_penalty(pre_b1, pre_b2):
    return 0


def parking_penalty(pre_b1, pre_b2):
    penalty = 0
    if pre_b1 > 10:
        penalty += 4
    if pre_b2 > 10:
        penalty += 4
    return penalty


def print_policy(policy, title):
    print(f"\n{title}")
    for s1 in range(0, MAX_BIKES + 1, 5):
        row = []
        for s2 in range(0, MAX_BIKES + 1, 5):
            row.append(f"{policy[s1, s2]:>2}")
        print(f"s1={s1:02}: {' '.join(row)}")

def print_values(values, title):
    print(f"\n{title}")
    print("Sample value function (every 5 states):")
    for s1 in range(0, MAX_BIKES + 1, 5):
        row = []
        for s2 in range(0, MAX_BIKES + 1, 5):
            row.append(f"{values[s1, s2]:7.2f}")
        print(f"s1={s1:02}: {' '.join(row)}")

def print_full_policy(policy, title):
    print(f"\n{title} - Full Policy Table")
    print("s1\\s2", end="")
    for s2 in range(MAX_BIKES + 1):
        if s2 % 5 == 0:
            print(f"{s2:>4}", end="")
    print()
    for s1 in range(MAX_BIKES + 1):
        if s1 % 5 == 0:
            print(f"{s1:3}", end="")
            for s2 in range(MAX_BIKES + 1):
                if s2 % 5 == 0:
                    print(f"{policy[s1, s2]:>4}", end="")
            print()

if __name__ == "__main__":
    print("="*60)
    print("PART 2: Baseline Policy Iteration")
    print("="*60)
    base_policy, base_values = policy_iteration(base_transfer_cost, no_parking_penalty)
    print_policy(base_policy, "Optimal policy (baseline Jack's/Gbike)")
    print_values(base_values, "Value function (baseline)")

    print("\n" + "="*60)
    print("PART 3: Modified Policy Iteration (Free Transfer + Parking Penalty)")
    print("="*60)
    modified_policy, modified_values = policy_iteration(modified_transfer_cost, parking_penalty)
    print_policy(modified_policy, "Optimal policy (free shuttle + parking fee)")
    print_values(modified_values, "Value function (modified)")
    
    print("\n" + "="*60)
    print("COMPARISON: Key State Differences")
    print("="*60)
    print("State (s1, s2) | Baseline Action | Modified Action | Baseline Value | Modified Value")
    print("-" * 80)
    for s1 in [0, 5, 10, 15, 20]:
        for s2 in [0, 5, 10, 15, 20]:
            if base_policy[s1, s2] != modified_policy[s1, s2]:
                print(f"({s1:2}, {s2:2})        | {base_policy[s1, s2]:>13} | {modified_policy[s1, s2]:>14} | {base_values[s1, s2]:>13.2f} | {modified_values[s1, s2]:>14.2f}")
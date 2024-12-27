# Define the states and actions
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
actions = ['up', 'right', 'down', 'left']

# Define the transition model
transitions = {
    's1': {'up': 's1', 'right': 's2', 'down': 's4', 'left': 's1'},
    's2': {'up': 's2', 'right': 's3', 'down': 's5', 'left': 's1'},
    's3': {'up': 's3', 'right': 's3', 'down': 's6', 'left': 's2'},
    's4': {'up': 's1', 'right': 's5', 'down': 's7', 'left': 's4'},
    's5': {'up': 's2', 'right': 's6', 'down': 's8', 'left': 's4'},
    's6': {'up': 's3', 'right': 's6', 'down': 's9', 'left': 's5'},
    's7': {'up': 's4', 'right': 's7', 'down': 's7', 'left': 's7'},
    's8': {'up': 's5', 'right': 's8', 'down': 's8', 'left': 's8'},
    's9': {'up': 's6', 'right': 's9', 'down': 's9', 'left': 's9'},
}

# Define the reward function
def reward(s, a, next_s):
    return 1 if next_s == 's9' else 0

# Set the discount factor and convergence threshold
gamma = 0.9
threshold = 1e-6

# Initialize the policy, e.g., always choose 'right'
policy = {s: 'right' for s in states if s != 's9'}

# Initialize the value function
V = {s: 0 for s in states}

# Function to perform policy evaluation
def policy_evaluation(V, policy, transitions, reward, gamma, threshold):
    while True:
        delta = 0
        V_new = V.copy()
        for s in states:
            if s == 's9':
                continue  # Terminal state
            a = policy[s]
            next_s = transitions[s][a]
            v = V[s]
            V_new[s] = reward(s, a, next_s) + gamma * V[next_s]
            delta = max(delta, abs(V_new[s] - v))
        V = V_new
        if delta < threshold:
            break
    return V

# Function to perform policy improvement
def policy_improvement(V, policy, transitions, reward, gamma):
    policy_stable = True
    for s in states:
        if s == 's9':
            continue  # Terminal state
        old_a = policy[s]
        # Compute value for all possible actions
        action_values = {}
        for a in actions:
            next_s = transitions[s][a]
            action_values[a] = reward(s, a, next_s) + gamma * V[next_s]
        # Choose the action with the highest value
        best_a = max(action_values, key=action_values.get)
        if old_a != best_a:
            policy_stable = False
        policy[s] = best_a
    return policy_stable

# Perform Policy Iteration
while True:
    # Policy Evaluation
    V = policy_evaluation(V, policy, transitions, reward, gamma, threshold)
    # Policy Improvement
    policy_stable = policy_improvement(V, policy, transitions, reward, gamma)
    if policy_stable:
        break

# Print the optimal value function and policy
print("Optimal Value Function:")
for s in states:
    print(f"V({s}) = {V[s]:.3f}")

print("\nOptimal Policy:")
for s in states:
    if s == 's9':
        continue
    print(f"π({s}) = {policy[s]}")

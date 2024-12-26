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
def reward(s, a):
    next_s = transitions[s][a]
    return 1 if next_s == 's9' else 0

# Set the discount factor and convergence threshold
gamma = 0.9
threshold = 1e-6

# Initialize the value function
V = {s: 0 for s in states}

# Value Iteration loop
while True:
    delta = 0
    for s in states:
        if s == 's9':
            continue  # Terminal state, V(s9) = 0
        v = V[s]
        # Compute the value for each action and choose the maximum
        max_val = max(reward(s, a) + gamma * V[transitions[s][a]] for a in actions)
        V[s] = max_val
        delta = max(delta, abs(V[s] - v))
    if delta < threshold:
        break

# Extract the optimal policy
policy = {}
for s in states:
    if s == 's9':
        continue  # Terminal state, no action
    best_a = None
    best_val = -float('inf')
    for a in actions:
        next_s = transitions[s][a]
        val = reward(s, a) + gamma * V[next_s]
        if val > best_val:
            best_val = val
            best_a = a
    policy[s] = best_a

# Print the optimal value function and policy
print("Optimal Value Function:")
for s in states:
    print(f"V({s}) = {V[s]:.3f}")

print("\nOptimal Policy:")
for s in states:
    if s == 's9':
        continue
    print(f"π({s}) = {policy[s]}")

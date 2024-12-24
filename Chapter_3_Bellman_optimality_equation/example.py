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
def reward(s, a, s_next):
    return 1 if s_next == 's9' else 0

# Set the discount factor and convergence threshold
gamma = 0.9
threshold = 1e-6

# Initialize value function
V = {s: 0 for s in states}

# Value iteration
while True:
    V_new = V.copy()
    for s in states:
        if s == 's9':
            V_new[s] = 0
            continue
        max_value = -float('inf')
        for a in actions:
            s_next = transitions[s][a]
            val = reward(s, a, s_next) + gamma * V[s_next]
            if val > max_value:
                max_value = val
        V_new[s] = max_value
    # Check for convergence
    max_change = max(abs(V_new[s] - V[s]) for s in states)
    if max_change < threshold:
        break
    V = V_new

# Print the optimal value function
for s in states:
    print(f"V({s}) = {V[s]:.3f}")

#基于模型的传统策略迭代（policy iteration)


import numpy as np

def policy_evaluation_model_based(P, R, policy, gamma=0.9, theta=1e-6):
    V = np.zeros(len(policy))
    while True:
        delta = 0
        for s in range(len(policy)):
            v = V[s]
            V[s] = sum([P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                        for a in range(len(policy[s]))
                        for next_s in range(len(V)) if P[s][a][next_s] > 0])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement_model_based(P, R, V, gamma=0.9):
    policy = np.zeros(len(V), dtype=int)
    for s in range(len(V)):
        Q_s = np.array([sum([P[s][a][next_s] * (R[s][a][next_s] + gamma * V[next_s])
                             for next_s in range(len(V)) if P[s][a][next_s] > 0])
                        for a in range(len(P[s]))])
        policy[s] = np.argmax(Q_s)
    return policy

def policy_iteration_model_based(P, R, gamma=0.9, theta=1e-6):
    n_states = len(P)
    policy = np.random.choice(len(P[0]), size=n_states)  # 初始化随机策略
    while True:
        old_policy = np.copy(policy)
        V = policy_evaluation_model_based(P, R, policy, gamma, theta)
        policy = policy_improvement_model_based(P, R, V, gamma)
        if np.all(old_policy == policy):
            break
    return policy, V


# 2. 改进版使用蒙特卡洛方法进行策略评估
# 每次访问蒙特卡洛估计（Every_Visit Monte Carlo)
def every_visit_monte_carlo(env, policy, episodes=500, gamma=0.9):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(episodes):
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(env.action_space.n, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            returns_sum[(state_t, action_t)] += G
            returns_count[(state_t, action_t)] += 1.0
            Q[state_t][action_t] = returns_sum[(state_t, action_t)] / returns_count[(state_t, action_t)]

    return Q

# 首次访问蒙特卡洛估计（First-visit Monte carlo
from collections import defaultdict

def first_visit_monte_carlo(env, policy, episodes=500, gamma=0.9):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(episodes):
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(env.action_space.n, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        state_action_in_episode = set([(x[0], x[1]) for x in episode])
        G = 0
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            if (state_t, action_t) in state_action_in_episode:
                returns_sum[(state_t, action_t)] += G
                returns_count[(state_t, action_t)] += 1.0
                Q[state_t][action_t] = returns_sum[(state_t, action_t)] / returns_count[(state_t, action_t)]
                state_action_in_episode.remove((state_t, action_t))

    return Q

# 3.策略改进（适合所有版本）
def policy_improvement(Q, nA):
    def policy_fn(state):
        A = np.ones(nA, dtype=float) / nA  # 初始化为均匀分布
        best_action = np.argmax(Q[state])
        A = np.eye(nA)[best_action]  # 更新为贪心策略
        return A
    return policy_fn

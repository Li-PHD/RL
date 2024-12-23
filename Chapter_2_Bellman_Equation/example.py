import numpy as np

# 定义状态空间
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']

# 定义动作空间
actions = ['a1', 'a2', 'a3', 'a4']  # 上、右、下、左

# 定义状态转移函数（确定性）
def transition(state, action):
    state_map = {
        's1': {'a1': 's1', 'a2': 's2', 'a3': 's4', 'a4': 's1'},
        's2': {'a1': 's2', 'a2': 's3', 'a3': 's5', 'a4': 's1'},
        's3': {'a1': 's3', 'a2': 's3', 'a3': 's6', 'a4': 's2'},
        's4': {'a1': 's1', 'a2': 's5', 'a3': 's7', 'a4': 's4'},
        's5': {'a1': 's2', 'a2': 's6', 'a3': 's8', 'a4': 's4'},
        's6': {'a1': 's3', 'a2': 's6', 'a3': 's9', 'a4': 's5'},
        's7': {'a1': 's4', 'a2': 's7', 'a3': 's7', 'a4': 's7'},
        's8': {'a1': 's5', 'a2': 's8', 'a3': 's8', 'a4': 's8'},
        's9': {'a1': 's6', 'a2': 's9', 'a3': 's9', 'a4': 's9'},
    }
    return state_map[state][action]

# 定义奖励函数
def reward(state):
    return 1 if state == 's9' else 0

# 折扣因子
gamma = 0.9

# 修正后的确定性策略
deterministic_policy = {
    's1': 'a2',  # 右
    's2': 'a2',  # 右
    's3': 'a3',  # 下
    's4': 'a3',  # 下
    's5': 'a4',  # 左
    's6': 'a3',  # 下
    's7': 'a4',  # 左
    's8': 'a4',  # 左
    's9': None,  # 目标状态
}

# 不确定性策略保持不变
stochastic_policy = {
    's1': {'a2': 0.5, 'a3': 0.5},  # 右: 0.5, 下: 0.5
    's2': {'a2': 0.5, 'a3': 0.5},  # 右: 0.5, 下: 0.5
    's3': {'a3': 1.0},  # 下: 1.0
    's4': {'a2': 0.5, 'a3': 0.5},  # 右: 0.5, 下: 0.5
    's5': {'a4': 1.0},  # 左: 1.0
    's6': {'a3': 1.0},  # 下: 1.0
    's7': {'a4': 1.0},  # 左: 1.0
    's8': {'a4': 1.0},  # 左: 1.0
    's9': {},  # 目标状态
}

# 计算状态值函数
def compute_state_value(policy, max_iter=1000, tol=1e-6):
    V = {s: 0 for s in states}  # 初始化状态值函数
    for _ in range(max_iter):
        V_new = V.copy()
        for s in states:
            if s == 's9':
                V_new[s] = 0  # 目标状态值为0
                continue
            if isinstance(policy[s], dict):  # 不确定性策略
                V_new[s] = sum(
                    policy[s].get(a, 0) * (reward(transition(s, a)) + gamma * V[transition(s, a)])
                    for a in actions
                )
            else:  # 确定性策略
                a = policy[s]
                V_new[s] = reward(transition(s, a)) + gamma * V[transition(s, a)]
        # 检查收敛性
        if max(abs(V_new[s] - V[s]) for s in states) < tol:
            break
        V = V_new
    return V

# 计算确定性策略的状态值
print("Deterministic Policy State Values:")
print(compute_state_value(deterministic_policy))

# 计算不确定性策略的状态值
print("Stochastic Policy State Values:")
print(compute_state_value(stochastic_policy))

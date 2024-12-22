import numpy as np

# 定义状态空间
# 3x3网格世界，状态编号为s1到s9
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']

# 定义动作空间
# 假设有4个动作：上(a1)、右(a2)、下(a3)、左(a4)
actions = ['a1', 'a2', 'a3', 'a4']

# 定义策略π
# 策略π是一个字典，键是状态，值是另一个字典，表示在该状态下采取每个动作的概率
policy = {
    's1': {'a1': 0.0, 'a2': 1.0, 'a3': 0.0, 'a4': 0.0},  # 在s1，100%向右走
    's2': {'a1': 0.0, 'a2': 0.5, 'a3': 0.5, 'a4': 0.0},  # 在s2，50%向右走，50%向下走
    's3': {'a1': 0.0, 'a2': 0.0, 'a3': 1.0, 'a4': 0.0},  # 在s3，100%向下走
    's4': {'a1': 0.5, 'a2': 0.0, 'a3': 0.5, 'a4': 0.0},  # 在s4，50%向上走，50%向下走
    's5': {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 1.0},  # 在s5，100%向左走
    's6': {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 0.0},  # 在s6，所有动作概率为0（可以省略）
    's7': {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 0.0},  # 在s7，所有动作概率为0（可以省略）
    's8': {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 0.0},  # 在s8，所有动作概率为0（可以省略）
    's9': {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 0.0},  # 在s9，所有动作概率为0（可以省略）
}

# 检查每个状态下的策略是否满足概率和为1
def check_policy_validity(policy):
    for state, action_probs in policy.items():
        prob_sum = sum(action_probs.values())
        if not np.isclose(prob_sum, 1.0):
            print(f"Warning: Policy for state {state} does not sum to 1. Sum is {prob_sum}")
        else:
            print(f"Policy for state {state} is valid.")

# 测试策略的有效性
check_policy_validity(policy)

# 模拟从某个状态开始，根据策略选择动作
def choose_action(state, policy):
    action_probs = policy[state]
    print(f"-----------------------------------------------action_probs: {action_probs}")
    actions = list(action_probs.keys())
    print(f"-----------------------------------------------actions: {actions}")
    probs = list(action_probs.values())
    print(f"-----------------------------------------------probs: {probs}")
    action = np.random.choice(actions, p=probs)  # 根据概率选择动作,用于从一个给定的数组中随机选择一个元素,并返回其索引。
    print(f"-----------------------------------------------action: {action}")
    return action

# 模拟从s1开始，根据策略选择动作
state = 's1'
action = choose_action(state, policy)
print(f"From state {state}, the chosen action is {action}")

# 模拟从s2开始，根据策略选择动作
state = 's2'
action = choose_action(state, policy)
print(f"From state {state}, the chosen action is {action}")

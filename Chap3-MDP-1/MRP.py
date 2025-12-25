import numpy as np

# ================马尔可夫奖励过程 <S, P, R, γ> ==================
np.random.seed(0)


# 状态转移矩阵 6*6
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

# 奖励函数
rewards = [-1, -2, -2, 10, 1, 0]
# 折扣因子
gamma = 0.5


# ================ 计算某条采样轨迹的回报 =================
# 计算一条状态序列的return
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = rewards[chain[i] - 1] + gamma * G
    return G


# 采样一个状态序列
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print(f"根据本序列得到的回报为:{G}")


# ================ 计算所有状态的价值函数 ========================
def compute(gamma, P, rewards, states_num):
    # 将rewards装换成列向量的形式
    rewards = np.array(rewards).reshape(-1, 1)
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


V = compute(gamma, P, rewards, 6)
print(f"MRP每个状态的价值分别为:{V}")

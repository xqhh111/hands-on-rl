import numpy as np


# =======================================环境========================================
class BernoulliBandit:
    """伯努利老虎机，输入K表示拉杆个数"""

    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 初始化每根拉杆获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最高的拉杆
        self.best_prob = self.probs[self.best_idx]  # 对应最大获奖概率
        self.K = K

    def step(self, k):
        # 玩家选择了k号拉杆后,按照概率给出的reward
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


# np.random.seed(1) # 设置随机种子
# K=10  # 10臂伯努利老虎机
# bandit_10_arm=BernoulliBandit(K)
# print(f"{K}臂老虎机")
# print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}号, 其获奖概率是{bandit_10_arm.best_prob:.4f}")


# ================================智能体=============================================
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 记录每根拉杆的尝试次数
        self.actions = []  # 列表。记录每一步动作
        self.regret = 0  # 当前步的累计懊悔
        self.regrets = []  # 列表。记录每一步的累计懊悔

    # 根据策略选择动作+根据动作获取奖励+更新期望奖励估值
    def run_one_step(self):
        # 具体选择动作的方法，由子类实现
        raise NotImplementedError

    def update_regret(self, k):
        # 更新动作带来的结果
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()  # 选择动作
            self.counts[k] += 1  # 更新尝试次数
            self.actions.append(k)  # 记录动作
            self.update_regret(k)  # 更新懊悔值

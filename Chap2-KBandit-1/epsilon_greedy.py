import numpy as np
from bandit_and_solver import Solver


class EpsilonGreedy(Solver):
    """epsilon-greddy算法，继承Solver类"""

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 乐观初始化，鼓励探索

    def run_one_step(self):
        # epsilon 决策阶段
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        # 环境给的奖励
        r = self.bandit.step(k)
        # 更新认知： 新估值=旧估值+学习率*（实际奖励-旧估值）
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0  # 记录总步数

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

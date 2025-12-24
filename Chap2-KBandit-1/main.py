import numpy as np
import matplotlib.pyplot as plt
from bandit_and_solver import BernoulliBandit
from epsilon_greedy import EpsilonGreedy, DecayingEpsilonGreedy
from UCB import UCB
from thompson_sampling import ThompsonSampling


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

# ==================== epsilon=0.01结果 =========================
# 设定环境中各机器臂的获奖概率 随机种子
np.random.seed(1)
# 定义环境
bandit_10_arm=BernoulliBandit(10)

# 设定智能体做决策相关 随机种子
np.random.seed(1)
# 定义智能体
epsilon_greedy_solver=EpsilonGreedy(bandit_10_arm,epsilon=0.01)
# 运行智能体
T=5000  # 轮数
epsilon_greedy_solver.run(T)

# 结果 epsilon=0.01
print("epsilon-greedy 累计懊悔：", epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver],["EpsilonGreedy"])

# =====================探索epsilon不同值的结果 ===========================
np.random.seed(0)
epsilons=[1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list=[
    EpsilonGreedy(bandit_10_arm,epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names=[
    f"epsilon={e}" for e in epsilons
]
for solver in epsilon_greedy_solver_list:
    solver.run(T)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# ===================== epsilon递减的情况 ===========================
np.random.seed(1)
decaying_epsilon_greedy_solver=DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(T)
print("decaying epsilon 累积懊悔值为:", decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

# ===================== UCB算法 (引入不确定性) =========================
np.random.seed(1)
coef=1
UCB_solver=UCB(bandit_10_arm,coef)
UCB_solver.run(T)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])


# ===================== Thompson Sampling 算法 =========================
np.random.seed(1)
thompson_sampling_solver=ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(T)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])
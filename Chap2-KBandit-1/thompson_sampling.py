import numpy as np
from bandit_and_solver import Solver

class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a=np.ones(self.bandit.K)  # 每根拉杆奖励为1次数
        self._b=np.ones(self.bandit.K)  # 每根拉杆奖励为0次数
        
    def run_one_step(self):
        samples=np.random.beta(self._a,self._b)
        k=np.argmax(samples)
        
        r=self.bandit.step(k)
        self._a[k]+=r
        self._b[k]+=(1-r)
        return k

"""
agt_myagt.py
"""
import numpy as np
import core


class MyAgt(core.coreAgt):
    """ オリジナルエージェントクラス """
    def __init__(self, alpha=0.01, epsilon=0.5):
        self.epsilon = epsilon
        # ------------------------- 編集ここから
        self.alpha = alpha
        self.Q = np.array([0.0, 0.0])
        # ------------------------- ここまで

    def select_action(self, obs):
        """ 観測に対して行動を出力 """
        # ------------------------- 編集ここから
        if np.random.rand() < self.epsilon:
            # epsilonの確率でランダム
            act = np.random.randint(0, 2)
        else:
            # 1- epsilonの確率
            if self.Q[0] == self.Q[1]:
                # Q値が同じだったらランダム
                act = np.random.randint(0, 2)
            else:
                # Q値が大きい方を選ぶ
                act = np.argmax(self.Q)
        # ------------------------- ここまで
        return act

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:
            return
        # ------------------------- 編集ここから
        self.Q[act] = (1 - self.alpha) * self.Q[act] \
            + self.alpha * rwd
        # ------------------------- ここまで
        return

    def get_Q(self, obs):
        """ 観測に対するQ値を出力 """
        # ------------------------- 編集ここから
        Q = self.Q
        # ------------------------- ここまで
        return Q

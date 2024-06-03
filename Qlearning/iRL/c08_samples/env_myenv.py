"""
env_myenv.py
"""
import cv2
import numpy as np
import core


class MyEnv(core.coreEnv):
    """ オリジナル環境クラス """
    def __init__(self):
        self.n_act = 2  # <--- 行動数を設定 (A)
        self.done = False
        # ------------------------- 編集ここから
        self.state = None
        # ------------------------- ここまで

    def reset(self):
        """ 状態を初期化 """
        self.done = False
        # ------------------------- 編集ここから
        self.state = 'start'
        obs = np.array([0])
        # ------------------------- ここまで
        return obs

    def step(self, act):
        """
        act = 0: ガラガラA
        act = 1: ガラガラB
        """
        # 最終状態の次の状態はリセット
        if self.done is True:
            obs = self.reset()
            return None, None, obs
        # ------------------------- 編集ここから
        if act == 0:  # ガラガラA  (A)
            if np.random.rand() < 0.5:
                rwd = 5.0
                self.state = 'A small'
            else:
                rwd = 10.0
                self.state = 'A big'
        else:  # ガラガラB (B)
            rwd = 7.0
            self.state = 'B'

        done = True          # (C)
        obs = np.array([9])  # (D)
        # ------------------------- ここまで
        self.done = done
        return rwd, done, obs

    def render(self):
        """ 状態に対応した画像を作成 """
        # ------------------------- 編集ここから

        if self.state == 'A small':
            x = 50
            col = (255, 255, 255)  # 白
        elif self.state == 'A big':
            x = 50
            col = (0, 255, 255)  # 黄色
        else:  # 'B'
            x = 150
            col = (0, 255, 0)  # 緑

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        if self.state != 'start':
            cv2.circle(img, (x, 50), 30, col, -1)
        # ------------------------- ここまで
        return img

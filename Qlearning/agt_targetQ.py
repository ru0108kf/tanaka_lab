"""
agt_targetQ.py
経験再生に
ターゲットネットワークを追加したQ学習のアルゴリズム
"""

import random
import numpy as np

# 自作モジュール
from agt_replayQ import ReplayQAgt


class TargetQAgt(ReplayQAgt):
    """
    経験再生にターゲットネットワークを取り入れた
    Q学習エージェントクラス
    """
    def __init__(self, **kwargs):
        # ターゲットインターバル  (A)
        self.target_interval = kwargs.pop('target_interval', 10)

        # 継承元のクラスの初期処理 (B)
        super().__init__(**kwargs)

        # ターゲットモデルの作成 (C)
        self.model_target = self._build_Qnet()
        self.time = 0  # タイムステップのカウンター (D)

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:
            return
        # ------------------------- 編集ここから
        # 経験の保存 (A)
        self.replay_memory.add((obs, act, rwd, done, next_obs))

        # 学習 (B)
        self._fit()

        # target_intervalの周期でQネットワークの重みをターゲットネットにコピー (C)
        if self.time % self.target_interval == 0 \
                and self.time > 0:
            self.model_target.set_weights(
                self.model.get_weights())

        self.time += 1
        # ------------------------- ここまで

    def _fit(self):
        """ 学習 """
        # 記憶された「経験」の量がバッチサイズに満たない場合は戻る (A)
        if len(self.replay_memory) < self.replay_batch_size:
            return

        # バッチサイズ個の経験を記憶からランダムに取得 (B)
        outs = self.replay_memory.sample(self.replay_batch_size)

        # 観測のバッチを入れる配列を準備 (C)
        obs = outs[0][0]  # 1番目の経験の観測
        obss = np.zeros(
            (self.replay_batch_size,) + obs.shape,
            dtype=int)

        # ターゲットのバッチを入れる配列を準備 (D)
        targets = np.zeros((self.replay_batch_size, self.n_act))

        # 経験ごとのループ (E)
        for i, out in enumerate(outs):
            # 経験を要素に分解 (F)
            obs, act, rwd, done, next_obs = out

            # obs に対するQネットワークの出力 yを得る (G)
            y = self.get_Q(obs, type='main')

            # target にyの内容をコピーする (H)
            target = y.copy()

            if done is False:
                # 最終状態でなかったら next_obsに対する next_yを
                # ターゲットネットから得る (I)
                next_y = self.get_Q(next_obs, type='target')

                # Q[obs][act]のtarget_act を作成
                target_act = rwd + self.gamma * max(next_y)
            else:
                # 最終状態の場合は報酬だけでtarget_actを作成
                target_act = rwd

            # targetのactの要素だけtarget_actにする
            target[act] = target_act

            # obsとtargetをバッチの配列に入れる
            obss[i, :] = obs
            targets[i, :] = target

        # obssと targets のバッチのペアを与えて学習
        self.model.fit(obss, targets, verbose=0, epochs=1)

    def get_Q(self, obs, type='main'):
        """ 観測に対するQ値を出力 """
        if type == 'main':
            # Qネットワークに観測obsを入力し出力を得る (A)
            Q = self.model.predict(obs.reshape(
                (1,) + self.input_size), verbose=0)[0, :]
        elif type == 'target':
            # ターゲットネットに観測obsを入力し出力を得る (B)
            Q = self.model_target.predict(obs.reshape(
                (1,) + self.input_size), verbose=0)[0, :]
        else:
            raise ValueError('get_Q のtype が不適切です')

        return Q


if __name__ == '__main__':
    # エージェントのインスタンス生成 (A)
    agt = TargetQAgt(n_act=3, input_size=(5,))

    # 行動選択 (B)
    obs = np.array([[1, 1, 1, 1, 1]])
    act = agt.select_action(obs)
    print('act', act)

    # 学習 (C)
    rwd = 1
    done = False
    next_obs = np.array([[1, 1, 1, 1, 2]])
    agt.learn(obs, act, rwd, done, next_obs)

    # モデル構造の表示 (D)
    print('モデルの構造')
    agt.model.summary()

    # 保存と読み込み (E)
    agt.save_weights('agt_data/test')

    # agt.load_weights('agt_data/test') (F)

    # モデルへの観測の入力 (G)
    y = agt.model.predict(obs)
    print('モデルの出力 y', y.reshape(-1))

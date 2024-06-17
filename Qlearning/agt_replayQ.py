"""
agt_replayQ.py
経験再生（Experience Replay） と
ニューラルネットを使ったQ学習のアルゴリズム
"""

import random
import numpy as np

# 自作モジュール
from agt_netQ import NetQAgt  # (A)


class ReplayMemory:  # (B)
    """ 経験を記録するクラス """
    def __init__(self, memory_size):
        self.memory_size = memory_size  # 記憶サイズ
        self.memory = []

    def __len__(self):  # (C)
        """ len()で、memoryの長さを返す """
        return len(self.memory)

    def add(self, experience):  # (D)
        """ 経験を記憶に追加する """
        # 経験の保存
        self.memory.append(experience)

        # 容量がオーバーしたら古いものを1つ捨てる
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample(self, data_length):  # (E)
        """ data_length分、ランダムにサンプルする """
        out = random.sample(self.memory, data_length)
        return out


class ReplayQAgt(NetQAgt):
    """ 経験再生とQネットワークを使ったQ学習エージェントクラス """
    def __init__(self, **kwargs):
        # int: 記憶サイズ (A)
        self.replay_memory_size = \
            kwargs.pop('replay_memory_size', 100)

        # int: 学習のバッチサイズ (B)
        self.replay_batch_size = \
            kwargs.pop('replay_batch_size', 20)

        # 経験再生クラスのインスタンス生成 (C)
        self.replay_memory = ReplayMemory(
            memory_size=self.replay_memory_size)

        # 継承元のクラスの初期処理 (D)
        super().__init__(**kwargs)

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:
            return
        # ------------------------- 編集ここから
        # 経験の保存 (A)
        self.replay_memory.add((obs, act, rwd, done, next_obs))

        # 学習 (B)
        self._fit()
        # ------------------------- ここまで

    def _fit(self):
        """ 学習の本体 """
        # 記憶された「経験」の量がバッチサイズに満たない場合は戻る (A)
        if len(self.replay_memory) < self.replay_batch_size:
            return

        # 経験メモリーからランダムにバッチサイズ分コピー (B)
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
            y = self.get_Q(obs)

            # target にyの内容をコピーする (H)
            target = y.copy()

            if done is False:
                # 最終状態でなかったら next_obsに対する next_yを得る (I)
                next_y = self.get_Q(next_obs)

                # Q[obs][act]のtarget_act を作成 (J)
                target_act = rwd + self.gamma * max(next_y)
            else:
                # 最終状態の場合は報酬だけでtarget_actを作成 (K)
                target_act = rwd

            # targetのactの要素だけtarget_actにする (L)
            target[act] = target_act

            # obsとtargetをバッチの配列に入れる (M)
            obss[i, :] = obs
            targets[i, :] = target

        # obssと targets のバッチのペアを与えて学習 (N)
        self.model.fit(obss, targets, verbose=0, epochs=1)


if __name__ == '__main__':
    # エージェントのインスタンス生成 (A)
    agt = ReplayQAgt(n_act=3, input_size=(5,))

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

    # 重みパラメータの読み込み (F)
    agt.load_weights('agt_data/test')

    # モデルへの観測の入力 (G)
    y = agt.model.predict(obs)
    print('モデルの出力 y', y.reshape(-1))

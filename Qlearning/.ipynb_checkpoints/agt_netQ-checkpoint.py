"""
agt_netQ.py
ニューラルネット（Qネットワーク）を使ったQ学習アルゴリズム
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import h5py

class NetQAgt():
    """ Qネットワークを使ったQ学習エージェントクラス """
    def __init__(               # 引数とデフォルト値の設定 (A)
            self,
            n_act=3,            # int: 行動の種類数（ネットワークの出力数）
            input_size=(4,),    # tuple of int: 入力サイズ
            n_dense=32,         # int: 中間層のニューロン数
            epsilon=0.1,        # float: 乱雑度
            gamma=0.9,          # float: 割引率
            filepath=None,      # str: 保存ファイル名
            ):
        """ 初期処理 """
        # 引数の設定は適時編集
        self.epsilon = epsilon
        # ------------------------- 編集ここから

        # アトリビュートにパラメータを保存 (B)
        self.n_act = n_act
        self.input_size = input_size
        self.n_dense = n_dense
        self.gamma = gamma
        self.filepath = filepath

        # アトリビュートにモデルを保存 (C)
        self.model = self._build_Qnet()
        # ------------------------- ここまで

    def _build_Qnet(self):
        """ 指定したパラメータでQネットワークを構築 """
        # Qネットワークの構築 (A)
        model = Sequential([
            Flatten(input_shape=self.input_size),
            Dense(self.n_dense, activation='relu'),
            Dense(self.n_act, activation='linear'),
        ])

        # 勾配法のパラメータの定義 (B)
        model.compile(
            optimizer='adam',
            loss='mse',
        )
        return model

    def select_action(self, obs):
        """  観測に対して行動を出力 """
        # ------------------------- 編集ここから
        # 確率的に処理を分岐 (A)
        if np.random.rand() < self.epsilon:
            # ランダム行動 (B)
            act = np.random.randint(0, self.n_act)
        else:
            # obsに対するQ値のリストを取得 (C)
            Q = self.get_Q(obs)

            # Qを最大にする行動
            act = np.argmax(Q)
        # ------------------------- ここまで
        return act

    def get_Q(self, obs):
        """ 観測に対するQ値を出力 """
        # ------------------------- 編集ここから
        # 観測obsを入力し出力を得る (A)
        Q = self.model.predict(
            obs.reshape((1,) + self.input_size))[0, :]
        # ------------------------- ここまで
        return Q

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:
            return
        # ------------------------- 編集ここから

        # obs に対するQネットワークの出力yを得る (A)
        y = self.get_Q(obs)

        # target にyの内容をコピーする (B)
        target = y.copy()

        if done is False:
            # 最終状態でなかったら next_obsに対する next_yを得る(C)
            next_y = self.get_Q(next_obs)

            # Q[obs][act]のtarget_actを作成 (D)
            target_act = rwd + self.gamma * max(next_y)
        else:
            # 最終状態の場合は報酬だけでtarget_actを作成 (E)
            target_act = rwd

        # targetのactの要素だけtarget_actにする (F)
        target[act] = target_act

        # obsと target のペアを与えて学習 (G)
        self.model.fit(
            obs.reshape((1,) + self.input_size),
            target.reshape(1, -1),
            verbose=0, epochs=1)
        # ------------------------- ここまで
        return

    def save_weights(self, filepath=None):
        """ モデルの重みデータの保存 """
        # ------------------------- 編集ここから
        if filepath is None:
            filepath = self.filepath
        self.model.save(filepath + '.keras', overwrite=True)
        # ------------------------- ここまで

    def load_weights(self, filepath=None):
        """ モデルの重みデータの読み込み """
        # ------------------------- 編集ここから
        if filepath is None:
            filepath = self.filepath
        self.model = tf.keras.models.load_model(filepath + '.keras')
        # ------------------------- ここまで


if __name__ == '__main__':
    # エージェントのインスタンス生成 (A)
    agt = NetQAgt(n_act=3, input_size=(5,))

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

    # 重みパラメータの保存 (E)
    agt.save_weights('agt_data/test')

    # 重みパラメータの読み込み (F)
    agt.load_weights('agt_data/test')

    # モデルへの観測の入力 (G)
    y = agt.model.predict(obs)
    print('モデルの出力 y', y.reshape(-1))

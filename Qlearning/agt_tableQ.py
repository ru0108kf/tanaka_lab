"""
agt_tableQ.py
Qテーブルを使ったQ学習アルゴリズム
"""
import sys
import pickle
import numpy as np


class TableQAgt():
    """ Qテーブルを使ったQ学習エージェントクラス """
    def __init__(           # 引数とデフォルト値の設定 (A)
            self,
            n_act=2,            # int: 行動の種類数
            init_val_Q=0,       # float: Q値の初期値
            epsilon=0.1,        # float: 乱雑度
            alpha=0.1,          # float: 学習率
            gamma=0.9,          # float: 割引率
            max_memory=500,     # int: 記憶する最大の観測数
            filepath=None,      # str: 保存用ファイル名
            ):
        """ 初期処理 """
        # 引数の設定は適時編集
        self.epsilon = epsilon
        # ------------------------- 編集ここから
        self.n_act = n_act
        # エージェントのハイパーパラメータ (B)
        self.init_val_Q = init_val_Q
        self.gamma = gamma
        self.alpha = alpha

        # 保存ファイル名 (C)
        self.filepath = filepath

        # Qテーブル関連 (D)
        self.Q = {}     # Qテーブル
        self.len_Q = 0  # Qテーブルに登録した観測の数
        self.max_memory = max_memory
        # ------------------------- ここまで

    def select_action(self, obs):
        """ 観測に対して行動を出力 """
        # ------------------------- 編集ここから
        # obsを文字列に変換 (A)
        obs = str(obs)

        # obs が登録されていなかったら初期値を与えて登録 (B)
        self._check_and_add_observation(obs)

        # 確率的に処理を分岐 (C)
        if np.random.rand() < self.epsilon:
            # epsilon の確率(D)
            act = np.random.randint(0, self.n_act)  # ランダム行動
        else:
            # 1-epsilon の確率(E)
            act = np.argmax(self.Q[obs])  # Qを最大にする行動
        # ------------------------- ここまで
        return act

    def _check_and_add_observation(self, obs):
        """ obs が登録されていなかったら初期値を与えて登録 """
        if obs not in self.Q:  # (A)
            self.Q[obs] = [self.init_val_Q] * self.n_act  # (B)
            self.len_Q += 1  # (C)
            if self.len_Q > self.max_memory:  # (D)
                print(f'観測の登録数が上限 ' +
                      f'{self.max_memory:d} に達しました。')
                sys.exit()
            if (self.len_Q < 100 and self.len_Q % 10 == 0) or \
                    (self.len_Q % 100 == 0):  # (E)
                print(f'the number of obs in Q-table' +
                      f' --- {self.len_Q:d}')

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:  # rwdがNoneだったら戻る(A)
            return
        # ------------------------- 編集ここから
        # obs, next_obs を文字列に変換 (B)
        obs = str(obs)
        next_obs = str(next_obs)

        # next_obs が登録されていなかったら初期値を与えて登録 (C)
        self._check_and_add_observation(next_obs)

        # 学習のターゲットを作成 (D)
        if done is True:
            target = rwd
        else:
            target = rwd + self.gamma * max(self.Q[next_obs])

        # Qをターゲットに近づける (E)
        self.Q[obs][act] = \
            (1 - self.alpha) * self.Q[obs][act] + \
            self.alpha * target
        # ------------------------- ここまで

    def get_Q(self, obs):
        """ 観測に対するQ値を出力 """
        # ------------------------- 編集ここから
        obs = str(obs)
        if obs in self.Q:   # obsがQにある (A)
            val = self.Q[obs]
            Q = np.array(val)
        else:               # obsがQにない (B)
            Q = None
        # ------------------------- ここまで
        return Q

    def save_weights(self, filepath=None):
        """ 方策のパラメータの保存 """
        # ------------------------- 編集ここから
        # Qテーブルの保存
        if filepath is None:
            filepath = self.filepath + '.pkl'
        with open(filepath, mode='wb') as f:
            pickle.dump(self.Q, f)
        # ------------------------- ここまで

    def load_weights(self, filepath=None):
        """ 方策のパラメータの読み込み """
        # ------------------------- 編集ここから
        # Qテーブルの読み込み
        if filepath is None:
            filepath = self.filepath + '.pkl'
        with open(filepath, mode='rb') as f:
            self.Q = pickle.load(f)
        # ------------------------- ここまで


if __name__ == '__main__':
    # 学習のステップ数 (A)
    n_step = 5000

    # コマンドライン引数 (B)
    argvs = sys.argv
    if len(argvs) > 1:
        n_step = int(argvs[1])
    print(f'{n_step:d}ステップの学習シミュレーション開始')

    # 環境の準備 (C)
    from env_corridor import CorridorEnv
    env = CorridorEnv()

    # 環境のパラメータの与え方の例
    """
    env = CorridorEnv(
        field_length=6,
        crystal_candidate=(2, 3, 4, 5),
        rwd_fail=-1,
        rwd_move=0,
        rwd_crystal=5,
    )
    """

    # エージェントの準備 (D)
    agt = TableQAgt(
        alpha=0.2,
        gamma=1,
        epsilon=0.5,
        )

    # 学習シミュレーション (E)
    obs = env.reset()
    for t in range(n_step):
        # エージェントが行動を選ぶ (F)
        act = agt.select_action(obs)

        # 環境が報酬と次の観測を決める (G)
        rwd, done, next_obs = env.step(act)

        # エージェントが学習する (H)
        agt.learn(obs, act, rwd, done, next_obs)

        # next_obsを次の学習のために保持 (I)
        obs = next_obs

    # 学習後のQ値の表示のための入力観測 (J)
    obss = [
        '[1 0 2 0]',
        '[0 1 2 0]',
        '[0 0 1 0]',
        '[0 0 2 1]',
        '[1 0 0 2]',
        '[0 1 0 2]',
        '[0 0 1 2]',
        '[0 0 0 1]',
    ]

    # 学習後のQ値の表示 (K)
    print('')
    print('学習後のQ値')
    for obs in obss:
        q_vals = agt.get_Q(obs)
        if q_vals is not None:
            msg = (
                f'{obs}: ' +
                f'{agt.Q[obs][0]: .2f}, ' +
                f'{agt.Q[obs][1]: .2f}'
            )
            print(msg)
        else:
            print(f'{obs}:')

    # 学習結果を見せるためのシミュレーションの準備(L)
    import cv2
    agt.epsilon = 0

    # 強化学習情報の初期化 (M)
    t = 0
    obs = env.reset()
    act = None
    rwd = None
    done = None
    next_obs = None

    # 開始メッセージを表示 (N)
    print('')
    print('学習なしシミュレーション開始')

    # 強化学習情報表示関数の定義 (O)
    def show_info(t, act, rwd, done, obs, isFirst=None):
        if rwd is None:
            if isFirst:
                tt = t
            else:
                tt = t + 1
            print('')
            print(f'x({tt:d})={str(obs):s}')
        else:
            msg = (
                f'a({t:d})={act:d}, ' +
                f'r({t:d})={rwd: .2f}, ' +
                f'done({t:d})={done:}, ' +
                f'x({t + 1:d})={str(obs):s}'
            )
            print(msg)

    # 強化学習情報表示 (P)
    show_info(t, act, rwd, done, obs, isFirst=True)

    # 学習なしシミュレーション (Q)
    while True:
        # 画面表示 (R)
        image = env.render()
        cv2.imshow('agt', image)

        # キーの受付と終了処理 (S)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        # エージェントの行動選択 (T)
        act = agt.select_action(obs)

        # 環境の更新 (U)
        rwd, done, obs = env.step(act)

        # 強化学習情報表示 (V)
        show_info(t, act, rwd, done, obs)
        t += 1

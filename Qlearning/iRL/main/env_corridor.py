"""
env_corridor.py
コリドータスク
"""
import numpy as np  # ベクトル・行列演算ライブラリ (A)
import cv2          # 画像作成・画像表示・キー操作用のライブラリ (B)

# 自作モジュール
import core         # core.py (C)
import myutil       # myutil.py (D)

PATH_BLANK = 'image/blank.png'            # 空白（床）(E)
PATH_ROBOT = 'image/robo_right.png'       # ロボット画像 (F)
PATH_CRYSTAL = 'image/crystal_small.png'  # クリスタル画像 (G)


class CorridorEnv(core.coreEnv):
    """ コリドータスクの環境クラス """
    # 内部表現のID (A)
    ID_blank = 0  # 空白
    ID_robot = 1  # ロボット
    ID_crystal = 2  # クリスタル

    def __init__(                   # (B)
            self,
            field_length=4,         # int: フィールドの長さ
            crystal_candidate=(2, 3),  # tuple of int: ゴールの位置
            rwd_fail=-1.0,            # 失敗した時の報酬（ペナルティ）
            rwd_move=-1.0,            # 進んだ時の報酬（コスト）
            rwd_crystal=5.0,          # クリスタルを得た時の報酬
            ):
        """ 初期処理 """
        # 引数の設定は適時編集
        self.n_act = 2  # <--- 行動数を設定 (C)
        self.done = False
        # ------------------------- 編集ここから
        """ インスタンス生成時の処理 """
        # タスクパラメータ (D)
        self.field_length = field_length
        self.crystal_candidate = crystal_candidate
        self.rwd_fail = rwd_fail
        self.rwd_move = rwd_move
        self.rwd_crystal = rwd_crystal

        # 内部状態の変数 (E)
        self.robot_pos = None       # ロボットの位置
        self.crystal_pos = None     # クリスタルの位置
        self.robot_state = None     # render 用

        # 画像の読み込み (F)
        self.img_robot = cv2.imread(PATH_ROBOT)
        self.img_crystal = cv2.imread(PATH_CRYSTAL)
        self.img_blank = cv2.imread(PATH_BLANK)
        self.unit = self.img_robot.shape[0]
        # ------------------------- ここまで

    def reset(self):
        """ 状態を初期化 """
        self.done = False  # (A)
        # ------------------------- 編集ここから
        # ロボットを通常状態に戻す (B)
        self.robot_state = 'normal'

        # ロボットの位置を開始位置へ戻す (C)
        self.robot_pos = 0

        # クリスタルの位置をランダムに決める (D)
        idx = np.random.randint(len(self.crystal_candidate))
        self.crystal_pos = self.crystal_candidate[idx]

        # ロボットとクリスタルの位置から観測を作る (E)
        obs = self._make_obs()
        # ------------------------- ここまで
        return obs

    def _make_obs(self):
        """ 状態から観測を作成 """
        # 最終状態判定がTrueだったら 9999 を出力 (A)
        if self.done is True:
            obs = np.array([9] * self.field_length)
            return obs

        # ロボットとクリスタルの位置から観測を作成 (B)
        obs = np.ones(self.field_length, dtype=int) \
            * CorridorEnv.ID_blank
        obs[self.crystal_pos] = CorridorEnv.ID_crystal
        obs[self.robot_pos] = CorridorEnv.ID_robot

        return obs

    def step(self, act):
        """ 状態を更新 """
        # 最終状態の次の状態はリセット(A)
        if self.done is True:
            obs = self.reset()
            return None, None, obs
        # ------------------------- 編集ここから
        # 行動act に対して状態を更新する (B)
        if act == 0:  # 拾う (C)
            if self.robot_pos == self.crystal_pos:
                # クリスタルの場所で「拾う」を選んだ
                rwd = self.rwd_crystal
                done = True
                self.robot_state = 'success'
            else:
                # クリスタル以外の場所で「拾う」を選んだ
                rwd = self.rwd_fail
                done = True
                self.robot_state = 'fail'
        else:  # act==1 進む (D)
            next_pos = self.robot_pos + 1
            if next_pos >= self.field_length:
                # 右端で「進む」を選んだ
                rwd = self.rwd_fail
                done = True
                self.robot_state = 'fail'
            else:
                # 右端より前で「進む」を選んだ
                self.robot_pos = next_pos
                rwd = self.rwd_move
                done = False
                self.robot_state = 'normal'
        # ------------------------- ここまで
        self.done = done  # (E)
        # ------------------------- 編集ここから
        obs = self._make_obs()  # obsを作成(F)
        # ------------------------- ここまで
        return rwd, done, obs

    def render(self):
        """ 状態に対応した画像を作成 """
        # ------------------------- 編集ここから
        # 画像サイズ width x height を決める(A)
        width = self.unit * self.field_length
        height = self.unit

        # カラー画像用の3次元配列変数を準備 (B)
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 空白部分（床）の描画 (C)
        for i_x in range(self.field_length):
            img = myutil.copy_img(
                img, self.img_blank,
                self.unit * i_x, 0,
                )

        # クリスタルの描画 (D)
        if self.robot_state != 'success':
            img = myutil.copy_img(
                img, self.img_crystal,
                self.unit * self.crystal_pos, 0,
                isTrans=True,
                )

        # ロボットの描画 (E)
        img = self._draw_robot(img)
        # ------------------------- ここまで
        return img

    def _draw_robot(self, img):
        """ ロボットを描く """
        # ロボットの頭の色の定義 (A)
        col_target = (224, 224, 224)  # ロボットの頭の色
        col_fail = (0, 0, 255)    # 赤
        col_success = (0, 200, 0)  # 緑

        # ロボット元画像をコピー (B)
        img_robot = self.img_robot.copy()

        # 頭の色を赤か緑に塗る (C)
        idx = np.where((img_robot == col_target).all(axis=2))
        if self.robot_state == 'fail':
            img_robot[idx] = col_fail
        elif self.robot_state == 'success':
            img_robot[idx] = col_success

        # ロボット画像の貼り付け (D)
        x0 = np.array(self.robot_pos) * self.unit
        img = myutil.copy_img(img, img_robot,
                              x0, 0, isTrans=True)

        return img


if __name__ == '__main__':
    # 操作方法の表示 (A)
    msg = (
        '\n' +
        '---- 操作方法 -------------------------------------\n'
        '[f] 右に進む\n' +
        '[d] 拾う\n' +
        '[q] 終了\n' +
        'クリスタルを拾うと成功\n' +
        '---------------------------------------------------'
    )
    print(msg)

    # 環境の準備 (B)
    env = CorridorEnv()

    # 環境のパラメータの与え方例
    """
    env = CorridorEnv(
        field_length=6,
        crystal_candidate=(2, 3, 4, 5),
        rwd_fail=-1,
        rwd_move=0,
        rwd_crystal=10,
    )
    """

    # 強化学習情報の初期化 (C)
    t = 0
    obs = env.reset()
    act = None
    rwd = None
    done = None

    # 開始の表示 (D)
    print('')
    print('あなたのプレイ開始')

    # 強化学習情報表示の関数定義 (E)
    def show_info(t, act, rwd, done, obs, isFirst=False):
        """ 強化学習情報の表示 """
        if rwd is None:  # (F)
            if isFirst:
                tt = t
            else:
                tt = t + 1
            print('')
            print(f'x({tt:d})={str(obs):s}')
        else:  # (G)
            msg = (
                f'a({t:d})={act:d}, ' +
                f'r({t:d})={rwd: .2f}, ' +
                f'done({t:d})={done:}, ' +
                f'x({t + 1:d})={str(obs):s}'
            )
            print(msg)

    # 強化学習情報表示 (H)
    show_info(t, act, rwd, done, obs, isFirst=True)

    # シミュレーション (I)
    while True:
        # 画面表示 (J)
        image = env.render()
        cv2.imshow('you', image)

        # キーの受付と終了処理 (K)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        # あなたの行動選択 (L)
        if key in [ord('d'), ord(' ')]:
            act = 0  # 拾う
        elif key == ord('f'):
            act = 1  # 進む
        else:
            continue

        # 環境の更新 (M)
        rwd, done, obs = env.step(act)

        # 強化学習情報表示 (N)
        show_info(t, act, rwd, done, obs)
        t += 1

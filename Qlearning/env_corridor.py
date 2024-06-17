# ライブラリ
import numpy as np  # ベクトル・行列演算ライブラリ
import pygame       # 画像作成・画像表示・キー操作用のライブラリ
import sys

class CorridorEnv():
    """ コリドータスクの環境クラス """
    # 内部表現のID
    ID_blank = 0  # 空白
    ID_robot = 1  # ロボット
    ID_crystal = 2  # クリスタル

    def __init__(self,
            field_length=4,         # int: フィールドの長さ
            crystal_candidate=(2, 3),  # tuple of int: ゴールの位置
            rwd_fail=-1.0,            # 失敗した時の報酬（ペナルティ）
            rwd_move=-1.0,            # 進んだ時の報酬（コスト）
            rwd_crystal=5.0,          # クリスタルを得た時の報酬
            ):
        """ 初期処理 """
        # 行動の数
        self.n_act = 2
        # 最終状態判定
        self.done = False
        
        """ インスタンス生成時の処理 """
        # タスクパラメータ
        self.field_length = field_length
        self.crystal_candidate = crystal_candidate
        self.rwd_fail = rwd_fail
        self.rwd_move = rwd_move
        self.rwd_crystal = rwd_crystal

        # 内部状態の変数
        self.robot_pos = None       # ロボットの位置
        self.crystal_pos = None     # クリスタルの位置
        self.robot_state = None     # render 用
        
    def reset(self):
        """ 状態を初期化 """
        self.done = False
        
        # ロボットを通常状態に戻す
        self.robot_state = 'normal'

        # ロボットの位置を開始位置へ戻す
        self.robot_pos = 0

        # クリスタルの位置をランダムに決める
        idx = np.random.randint(len(self.crystal_candidate))
        self.crystal_pos = self.crystal_candidate[idx]

        # ロボットとクリスタルの位置から観測を作る
        obs = self._make_obs()
        
        return obs

    def _make_obs(self):
        """ 状態から観測を作成 """
        # 最終状態判定がTrueだったら 9999 を出力
        if self.done is True:
            obs = np.array([9] * self.field_length)
            return obs

        # ロボットとクリスタルの位置から観測を作成
        obs = np.ones(self.field_length, dtype=int) * CorridorEnv.ID_blank # 1 * 0
        obs[self.crystal_pos] = CorridorEnv.ID_crystal
        obs[self.robot_pos] = CorridorEnv.ID_robot

        return obs

    def step(self, act):
        """ 状態を更新 """
        # 最終状態の次の状態はリセット
        if self.done is True:
            obs = self.reset()
            return None, None, obs

        # 行動act に対して状態を更新する
        if act == 0:  # 拾う
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
        else:  # act==1 進む
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

        self.done = done 
        # obsを作成
        obs = self._make_obs()
        
        return rwd, done, obs
    
    def pygame_render(self):
        """ pygameの描画 """
        # 色の定義
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        GRAY = (128, 128, 128)
        
        # 画面の更新
        self.screen.fill(WHITE)

        # フィールドの描画
        for field in self.fields:
            pygame.draw.rect(self.screen, BLACK, field, 1)
            
        # クリスタルの描画
        pygame.draw.circle(self.screen, BLUE, self.fields[self.crystal_pos].center, self.unit // 4)    
        
        # ロボットの描画
        if self.robot_state == 'normal':
            robot_color = GRAY
        elif self.robot_state == 'success':
            robot_color = GREEN
        elif self.robot_state == 'fail':
            robot_color = RED
        triangle_size = self.unit // 4
        x, y = self.fields[self.robot_pos].center
        points = [(x + triangle_size, y), (x - triangle_size, y - triangle_size), (x - triangle_size, y + triangle_size)]
        pygame.draw.polygon(self.screen, robot_color, points)
    
    def pygame_init(self):
        """pygameの開始"""
        pygame.init()

        # スクリーンの設定
        self.unit = 100
        screen_size = (self.unit * self.field_length, self.unit)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Corridor Task - Robot Collecting Crystal")

        # フィールドの設定
        self.fields = [pygame.Rect(i * self.unit, 0, self.unit, self.unit) for i in range(self.field_length)]

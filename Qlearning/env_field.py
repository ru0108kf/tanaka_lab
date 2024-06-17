import sys
from enum import Enum, auto  # 列挙型クラス
import random
import numpy as np
import pickle
import pygame

class FieldEnv():
    """ フィールドタスクの環境クラス """
    # 内部表現のID
    ID_blank = 0
    ID_robot = 1
    ID_wall = 2
    ID_crystal = 3

    # 上、左、下、右、方向移動で座標変化
    dr = np.array([
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
        ])

    def __init__(               
            self,
            field_size=5,       # int: フィールドの1辺の長さ
            sight_size=3,       # int: 視野の長さ
            max_time=30,        # int: タイムリミット
            n_wall=1,           # int: 壁の数（map_type='random'用）
            n_crystal=2,        # int: クリスタルの数
            start_pos=(3, 3),   # tuple of int: スタート座標
            start_dir=0,        # int: スタート時の方向(0, 1, 2, or 3)
            rwd_hit_wall=-0.2,  # float: 壁に当たった時の報酬（ダメージ）
            rwd_move=-0.1,      # float: 動いたときの報酬（コスト）
            rwd_crystal=1.0,    # float: クリスタルを得た時の報酬
            map_type='random',  # str: 'random' or 'fixed_map'
            wall_observable=True,  # bool: 壁を観測に入れる
            ):
        """ 初期処理 """
        # 引数の設定は適時編集
        self.n_act = 3  # <--- 行動数を設定 
        self.done = False

        # タスクパラメータ 
        self.field_size = field_size
        self.sight_size = sight_size
        self.max_time = max_time
        self.n_wall = n_wall
        self.n_crystal = n_crystal
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.rwd_hit_wall = rwd_hit_wall
        self.rwd_move = rwd_move
        self.rwd_crystal = rwd_crystal
        self.map_type = map_type
        self.wall_observable = wall_observable

        # 内部状態の変数 (D)
        self.robot_pos = None       # ロボットの位置
        self.agt_dir = None         # ロボットの方向
        self.fieldmap = None        # クリスタルと壁の配置
        self.n_collected_crystal = 0  # 回収したクリスタルの数
        self.time = None            # タイムリミット用のカウント
        self.robot_state = None     # render 用

    def set_task_type(self, task_type):
        """ task_type を指定してparameterを一括設定 """
        if task_type == "no_wall":
            self.field_size = 5        # int: フィールドの1辺の長さ
            self.sight_size = 4        # int: 視野の長さ
            self.max_time = 15         # int: タイムリミット
            self.rwd_hit_wall = -0.2   # float: 壁に当たった時の罰
            self.rwd_move = -0.1       # float: 移動コスト
            self.rwd_crystal = 1.0     # float: クリスタルを得た時の報酬
            self.map_type = 'random'   # 指定したクリスタルと壁の数でマップを自動生成
            self.n_wall = 0            # int: 壁の数（'random'用）
            self.n_crystal = 1         # int: クリスタルの数（'random'用）
            self.start_pos = (2, 2)    # tuple of int: スタート座標
            self.start_dir = 0         # int: スタート時の方向(0, 1, 2, or 3)
            self.wall_observable = False  # bool: 壁を観測に入れる

        elif task_type == "fixed_wall":
            self.field_size = 5
            self.sight_size = 2
            self.max_time = 25
            self.rwd_hit_wall = -0.2
            self.rwd_move = -0.1
            self.rwd_crystal = 1.0
            self.map_type = 'fixed_map'  # マップを指定するタイプ
            # fixed_mapの場合は text_mapを指定
            # '-': blank, # 'c': crystal, 'w': wall
            self.text_map = [  # 行数と列数はfield_sizeと一致させる
                '--ww-',
                '-----',
                '-w--c',
                '-wcww',
                'ww--w',
                ]
            self.n_wall = None  # 使用せず
            self.n_crystal = 2
            self.start_pos = (0, 1)
            self.start_dir = 0
            self.wall_observable = True

        elif task_type == "random_wall":
            self.field_size = 7
            self.sight_size = 2
            self.max_time = 30
            self.rwd_hit_wall = -0.2
            self.rwd_move = -0.1
            self.rwd_crystal = 1.0
            self.map_type = 'random'
            self.n_wall = 4
            self.n_crystal = 4
            self.start_pos = (3, 3)
            self.start_dir = 0
            self.wall_observable = True
            
        else:
            raise ValueError('task_type が間違っています')

    def reset(self):
        """ 状態を初期化 """
        self.done = False
        self.robot_state = 'normal'   # render 用
        self.time = 0                 # タイムリミット用のカウントを0
        self.n_collected_crystal = 0  # 集めたクリスタルの数を0

        if self.map_type == 'random':  
            # マップをランダム生成
            for i in range(100):
                self._random_map()  # ランダムで壁とクリスタルを配置
                # スタート地点から全てのクリスタルが回収可能なら
                # ループを抜ける
                possible_crystal = self._map_check()
                if possible_crystal == self.n_crystal:
                    break
                if i == 99:
                    # 100回生成して全て解けないパターンだったらエラー終了
                    msg = 'マップが生成できません。' + \
                        '壁やクリスタルの数を減らしてください'
                    raise ValueError(msg)

        elif self.map_type == 'fixed_map':  
            # self.text_map の情報でマップを生成
            self._fixed_map()

        else:
            raise ValueError('map_type が間違っています')

        # 観測を生成  
        obs = self._make_obs()

        return obs

    def _make_obs(self):
        """
        状態から観測を作成
        reset(), step()で使用
        """

        # クリスタルを1で表すmap_crystalを生成
        map_crystal = self.fieldmap.copy()
        map_crystal[self.fieldmap == FieldEnv.ID_crystal] = 1
        map_crystal[self.fieldmap != FieldEnv.ID_crystal] = 0

        # crystal 観測用として、まずフィールドの3倍の大きのobs_crystalを作る
        f_s = self.field_size
        size = f_s * 3
        obs_crystal = np.zeros((size, size), dtype=int)

        # obs_crystalの中心にmap_crystalをコピー
        obs_crystal[f_s:f_s * 2, f_s:f_s * 2] = map_crystal

        # robot_posを中心とした視野の大きさの矩形を抜き出し、
        # クリスタルの観測obs_crystalを作成
        s_s = self.sight_size
        x_val = f_s + self.robot_pos[0]
        y_val = f_s + self.robot_pos[1]
        obs_crystal = obs_crystal[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]

        # ロボットの方向に合わせてobs_crystalを回転
        if self.agt_dir == 3:
            obs_crystal = np.rot90(obs_crystal)
        elif self.agt_dir == 2:
            for _ in range(2):
                obs_crystal = np.rot90(obs_crystal)
        elif self.agt_dir == 1:
            for _ in range(3):
                obs_crystal = np.rot90(obs_crystal)

        # 同様に壁の観測行列を作成
        if self.wall_observable is True:
            # 壁を1で表すmap_wallを生成
            map_wall = self.fieldmap.copy()
            map_wall[self.fieldmap == FieldEnv.ID_wall] = 1
            map_wall[self.fieldmap != FieldEnv.ID_wall] = 0

            # 壁観測用として、まずフィールドの3倍の大きさのobs_wallを作る
            obs_wall = np.ones((size, size), dtype=int)

            # obs_wallの中心にmap_wallをコピー
            obs_wall[f_s:f_s * 2, f_s:f_s * 2] = map_wall

            # robot_posを中心とした視野の大きさの矩形を抜き出し、
            # 壁の観測 obs_wallを作成
            obs_wall = obs_wall[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]

            # ロボットの方向に合わせてobs_wallを回転
            if self.agt_dir == 3:
                obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 2:
                for _ in range(2):
                    obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 1:
                for _ in range(3):
                    obs_wall = np.rot90(obs_wall)

            # obs_wall を obs_crystal に連結してobsとする
            obs = np.c_[obs_crystal, obs_wall]
        else:
            obs = obs_crystal

        if self.done is True:
            # 最終状態判定がTrueだったら全ての要素を9にする
            obs[:] = 9

        return obs

    def _fixed_map(self):
        """
        文字で表したマップをndarray行列に変換
        reset()で使用
        """

        # 文字によるマップデータself.text_mapを
        # 2次元ndarray型のself.fieldmapに変換
        myfield = []
        for mline in self.text_map:
            line = []
            id_val = None
            for i in mline:
                if i == 'w':
                    id_val = FieldEnv.ID_wall
                elif i == '-':
                    id_val = FieldEnv.ID_blank
                elif i == 'c':
                    id_val = FieldEnv.ID_crystal
                else:
                    raise ValueError('マップのコードに解釈できない文字が含まれています')
                line.append(id_val)
            myfield.append(line)
        self.fieldmap = np.array(myfield, dtype=int)

        # マップサイズを保存
        self.field_size = self.fieldmap.shape[0]

        # ロボットをスタート地点と方向にセット
        self.robot_pos = self.start_pos
        self.agt_dir = self.start_dir

    def _random_map(self):
        """
        ランダムなマップを生成
        reset()で使用
        """
        # ロボットをスター地点と方向にセット
        self.robot_pos = self.start_pos
        self.agt_dir = self.start_dir

        # フィールドを準備
        self.fieldmap = np.ones((self.field_size,) * 2, dtype=int) \
            * FieldEnv.ID_blank

        # クリスタルの位置を決める
        for _ in range(self.n_crystal):
            while True:
                x_val = random.randint(0, self.field_size - 1)
                y_val = random.randint(0, self.field_size - 1)
                if not(x_val == self.start_pos[0]
                        and y_val == self.start_pos[1]) \
                        and self.fieldmap[y_val, x_val] == FieldEnv.ID_blank:
                    self.fieldmap[y_val, x_val] = FieldEnv.ID_crystal
                    break

        # 壁の位置を決める
        for _ in range(self.n_wall):
            for i in range(100):
                x_val = random.randint(0, self.field_size - 1)
                y_val = random.randint(0, self.field_size - 1)
                if not(self.robot_pos[0] == x_val
                        and self.robot_pos[1] == y_val) \
                        and self.fieldmap[y_val, x_val] == FieldEnv.ID_blank:
                    self.fieldmap[y_val, x_val] = FieldEnv.ID_wall
                    break
            if i == 99:
                print('壁の数が多すぎてマップが作れません')
                sys.exit()

    def _map_check(self):
        """
        スタート地点から到達できるクリスタルの数を出力
        reset()で使用
        """
        field = self.fieldmap
        f_h, f_w = field.shape
        x_agt, y_agt = self.robot_pos

        # フィールドの周囲に4辺を付けたf_valというフィールドを準備
        f_val = np.zeros((f_h + 2, f_w + 2), dtype=int)
        f_val[1:-1, 1:-1] = field

        # スタート地点に対応するf_valの要素をenable=99 にする
        enable = 99
        f_val[y_agt + 1, x_agt + 1] = enable
        possible_crystal = 0

        # enableの隣のセルにクリスタルがないかを調べていく
        # 調べたセルはenable に書き換える
        while True:
            is_change = False
            for i_y in range(1, f_h + 1):
                for i_x in range(1, f_w + 1):
                    if f_val[i_y, i_x] == enable:
                        f_val, is_change, reached_crystal = \
                            self._count_update(f_val, i_x, i_y, enable)
                        possible_crystal += reached_crystal
            if is_change is False:
                break

        return possible_crystal

    def _count_update(self, f_val, i_x, i_y, enable):
        """
        フィールドマップ f_val に対して、
        i_x, i_y の4方向隣にクリスタルがないかを調査
        _map_check()で使用
        """
        d_agt = FieldEnv.dr
        is_change = False
        reached_crystal = 0
        for i in range(d_agt.shape[0]):
            # 上左下右の方向に対するループ
            if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] \
                    == FieldEnv.ID_blank or \
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] \
                    == FieldEnv.ID_crystal:
                # i_x, i_y から各方向の隣がblankかcrystalだったら
                if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] \
                        == FieldEnv.ID_crystal:
                    # i_x, i_y から各方向の隣がcrystalだったらクリスタルのカウントを+1
                    # その場所を enableで埋める
                    reached_crystal += 1
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    is_change = True
                elif f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] \
                        == FieldEnv.ID_blank:
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    # i_x, i_y から各方向の隣がblankだったら
                    # その場所を enableで埋める
                    is_change = True
                else:
                    raise ValueError('Err!')

        return f_val, is_change, reached_crystal

    def step(self, act):
        """ 状態を更新 """
        # 最終状態の次の状態はリセット
        if self.done is True:
            obs = self.reset()
            return None, None, obs

        self.robot_state = 'normal'
        if act == 0:
            # 進む 
            pos = self.robot_pos + FieldEnv.dr[self.agt_dir]
            if pos[0] < 0 or self.field_size <= pos[0] or \
                    pos[1] < 0 or self.field_size <= pos[1]:
                # フィールドの範囲外に進もうとした
                self.robot_state = 'hit_wall'
                rwd = self.rwd_hit_wall
                done = False

            elif self.fieldmap[pos[1], pos[0]] \
                    == FieldEnv.ID_crystal:
                # 進む先がクリスタルだった
                self.robot_state = 'success'
                self.fieldmap[pos[1], pos[0]] \
                    = FieldEnv.ID_blank
                rwd = self.rwd_crystal
                self.n_collected_crystal += 1
                if self.n_collected_crystal == self.n_crystal:
                    # 全てのクリスタルを回収したら最終状態
                    done = True
                    self.robot_pos = pos
                else:
                    # まだクリスタルが残っている
                    done = False
                    self.robot_pos = pos

            elif self.fieldmap[pos[1], pos[0]] \
                    == FieldEnv.ID_blank:
                # 進む先が空白（床）だった
                self.robot_pos = pos
                rwd = self.rwd_move
                done = False

            elif self.fieldmap[pos[1], pos[0]] \
                    == FieldEnv.ID_wall:
                # 進む先が壁だった
                self.robot_state = 'hit_wall'
                rwd = self.rwd_hit_wall
                done = False

            else:
                raise ValueError('Err!')

        elif act == 1:
            # 左に90度回転する 
            self.agt_dir = (self.agt_dir + 1) % 4
            rwd = self.rwd_move
            done = False

        elif act == 2:
            # 右に90度回転する 
            self.agt_dir = (self.agt_dir - 1) % 4
            rwd = self.rwd_move
            done = False

        else:
            raise ValueError('Err!')

        # タイムリミットに達したら最終状態 
        self.time += 1
        if self.time >= self.max_time:
            done = True
            if self.robot_state != 'success':
                rwd = self.rwd_hit_wall
                self.robot_state = 'timeover'

        self.done = done

        # self.done を使った処理
        obs = self._make_obs()  # 観測obsを作成 

        return rwd, done, obs
    
    def pygame_init(self):
        """ pygameの開始 """
        # pygameの開始
        pygame.init()
        
        # フォントの初期化
        self.font = pygame.font.SysFont('Arial', 13)  # フォントの初期化
        
        # スクリーンの設定
        self.unit = 80
        screen_size = (self.unit * self.field_size, self.unit * self.field_size + 50)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Field Task - Robot Collecting Crystal')
        
        # フィールドの設定
        self.fields = [pygame.Rect(i * self.unit, 0, self.unit, self.unit) for i in range(self.field_size**2)]
        
    def pygame_render(self, info_text):
        """ pygameの描画 """
        # 色の定義
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        GRAY = (128, 128, 128)
        LIGHT_GRAY = (200, 200, 200)

        # 画面の更新
        self.screen.fill(WHITE)
        
        # テキストの描画
        lines = info_text.split('\n')
        y = self.unit * self.field_size + 1
        for line in lines:
            text_surface = self.font.render(line, True, BLACK)  # テキストのレンダリング
            self.screen.blit(text_surface, (10, y))
            y += self.font.get_linesize()

        # フィールドの描画
        for i_y in range(self.field_size):
            for i_x in range(self.field_size):
                field = pygame.Rect(i_x * self.unit, i_y * self.unit, self.unit, self.unit)
                pygame.draw.rect(self.screen, BLACK, field, 1)
                
                # Check if the current cell is outside the robot's sight
                if abs(i_x - self.robot_pos[0]) > self.sight_size or abs(i_y - self.robot_pos[1]) > self.sight_size:
                    pygame.draw.rect(self.screen, LIGHT_GRAY, field)
                
                if self.fieldmap[i_y, i_x] == FieldEnv.ID_wall:
                    wall = pygame.Rect(i_x * self.unit, i_y * self.unit, self.unit, self.unit)
                    pygame.draw.rect(self.screen, BLACK, wall)
                elif self.fieldmap[i_y, i_x] == FieldEnv.ID_crystal:
                    pygame.draw.circle(self.screen, BLUE, field.center, self.unit // 4)

        # ロボットの描画
        if self.robot_state == 'normal':
            robot_color = GRAY
        elif self.robot_state == 'success':
            robot_color = GREEN
        elif self.robot_state == 'hit_wall' or self.robot_state == 'timeover':
            robot_color = RED

        x, y = self.robot_pos
        direction = self.agt_dir
        if direction == 0:  # Up
            points = [(x * self.unit + self.unit // 2, y * self.unit), 
                      (x * self.unit + self.unit, y * self.unit + self.unit), 
                      (x * self.unit, y * self.unit + self.unit)]
        elif direction == 1:  # Left
            points = [(x * self.unit, y * self.unit + self.unit // 2), 
                      (x * self.unit + self.unit, y * self.unit), 
                      (x * self.unit + self.unit, y * self.unit + self.unit)]
        elif direction == 2:  # Down
            points = [(x * self.unit + self.unit // 2, y * self.unit + self.unit), 
                      (x * self.unit + self.unit, y * self.unit), 
                      (x * self.unit, y * self.unit)]
        elif direction == 3:  # Right
            points = [(x * self.unit + self.unit, y * self.unit + self.unit // 2), 
                      (x * self.unit, y * self.unit), 
                      (x * self.unit, y * self.unit + self.unit)]

        pygame.draw.polygon(self.screen, robot_color, points)

        pygame.display.flip()  # 画面更新
        
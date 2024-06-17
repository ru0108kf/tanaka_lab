"""
trainer.py
環境とエージェントを組み合わせて動かす
学習シミュレーションと学習なしシミュレーション
"""
import time
import random
import numpy as np
import pygame


class Recorder:
    """ シミュレーションの履歴保存クラス """
    def __init__(self):
        """ 初期処理 """
        self.rwd_in_episode = 0     # 報酬/エピソード
        self.rwds_in_episode = []   # 報酬/エピソードのリスト
        self.step_in_episode = 0    # ステップ数/エピソード
        self.steps_in_episode = []  # ステップ数/エピソードのリスト

    def reset(self):
        """ 記録の初期化 """
        self.rwd_in_episode = 0
        self.rwds_in_episode = []
        self.step_in_episode = 0
        self.steps_in_episode = []

    def add(self, rwd, done):
        """ 記録の追加 """
        # 報酬がNoneだったらカウントしない
        if rwd is None:
            return

        # 報酬とステップのカウント
        self.rwd_in_episode += rwd
        self.step_in_episode += 1

        # 最終状態なら
        # 今のrwd_in_episodeとstep_in_episodeを
        # リストに保存してリセット
        if done is True:
            self.rwds_in_episode.append(self.rwd_in_episode)
            self.rwd_in_episode = 0
            self.steps_in_episode.append(self.step_in_episode)
            self.step_in_episode = 0

    def get_result(self):
        """ 報酬とステップのリストの平均を出力 """
        mean_rwds = np.mean(self.rwds_in_episode)
        self.rwds_in_episode = []
        mean_steps = np.mean(self.steps_in_episode)
        self.steps_in_episode = []
        return mean_rwds, mean_steps


class Trainer:
    """ トレーナークラス """
    def __init__(       # 引数とデフォルト値の設定
            self,
            agt=None,       # TableQAgt class: エージェント
            env=None,       # CorridorEnv class: 環境
            eval_env=None,  # CorridorEnv class: 評価用環境
            ):
        """ 初期処理 """
        # エージェントと環境のインスタンス
        self.agt = agt              # エージェント
        self.env = env              # 学習用
        self.eval_env = eval_env    # 評価用

        # 学習過程の記録関連
        self.hist_rwds = []         # rewards/episodeの履歴
        self.hist_steps = []        # steps/episodeの履歴
        self.hist_time = []         # 上記のステップ数
        self.hist_start_time = 0    # スタート時のステップ数
        self.obss = None            # Q値のチェック用の観測
        self.recorder = Recorder()  # データ保存用クラス

    def simulate(
            self,  # 以下、Noneはその設定をしない場合
            n_step=1000,         # ステップ数
            n_episode=None,      # エピソード数
            is_eval=True,        # 評価を行うか
            is_learn=True,       # 学習を行うか
            is_animation=False,  # アニメーション表示をするか
            seed=None,           # 乱数固定のシード値
            eval_interval=100,   # 評価間のステップ数
            eval_n_step=None,    # 評価のステップ数
            eval_n_episode=10,   # 評価のエピソード数
            eval_epsilon=0.0,    # 評価時の乱雑度
            eval_seed=None,      # 評価の乱数固定のシード値
            anime_delay=0.5,     # アニメのフレーム間の秒数
            obss=None,           # Q値のチェック用の観測
            eary_stop=None,      # 目標平均報酬(float)
            show_history=True,   # 途中経過表示をするか
            ):
        """ シミュレーション """
        # Qチェック用の観測をアトリビュートにセット
        self.obss = obss

        # 乱数の初期化
        # アニメーション時に評価の環境を再現するために必要
        if seed != None:
            random.seed(seed)

        # 学習シミュレーションの準備
        stime = time.time()
        timestep = 1
        episode = 1

        # ここから学習シミュレーション開始
        obs = self.env.reset()
        
        running = True
        
        # pygameの開始
        clock = pygame.time.Clock()
        self.env.pygame_init()
        msg = ""
        
        while running:
            # アニメーション描画
            if is_animation:
                self.env.pygame_render("no learning simulation .\n [q]quit")
                
                pygame.time.wait(int(anime_delay * 1000))
                
                # 操作
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                         
            else:
                self.env.pygame_render(f'time:{timestep}/episode:{episode} ' +'.\n' + msg)
                # 操作
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                
            # エージェントが行動を選ぶ
            act = self.agt.select_action(obs)

            # 環境が報酬と次の観測を決める
            rwd, done, next_obs = self.env.step(act)

            # エージェントが学習する
            if is_learn is True:
                self.agt.learn(obs, act, rwd, done, next_obs)

            # next_obs, next_done を次の学習のために保持
            obs = next_obs

            # 一定のステップ数で記録と評価と表示を行う (B)
            if is_eval is True:
                if timestep % eval_interval == 0:
                    # 評価を行う
                    eval_rwds, eval_steps = self._evaluation(
                        eval_n_step=eval_n_step,
                        eval_n_episode=eval_n_episode,
                        eval_epsilon=eval_epsilon,
                        eval_seed=eval_seed,
                        )

                    # 記録
                    self.hist_rwds.append(eval_rwds)
                    self.hist_steps.append(eval_steps)
                    self.hist_time.append(
                        self.hist_start_time + timestep)

                    # 途中経過表示
                    if show_history:
                        ptime = int(time.time() - stime)
                        msg = f'{self.hist_start_time + timestep} steps, ' \
                            + f'{ptime} sec --- ' + '\n' \
                            + f'rewards/episode {eval_rwds: .2f}, ' \
                            + f'steps/episode {eval_steps: .2f}'

                    # 評価eval_rwdsが指定値eary_stopよりも上だったら終了
                    if eary_stop is not None:
                        if eval_rwds > eary_stop:
                            msg = f'eary_stop eval_rwd {eval_rwds:.2f}' \
                                + f' > th {eary_stop:.2f}'
                            print(msg)
                            break

            # 終了判定
            timestep += 1
            
            episode += (done is True)
            if n_step is not None:
                if timestep >= n_step + 1:
                    break
            if n_episode is not None:
                if episode >= n_episode + 1:
                    break

        # シミュレーション終了後にQ値を表示
        self._show_Q()
        pygame.quit()

    def _evaluation(
            self,
            eval_n_step,
            eval_n_episode,
            eval_epsilon,
            eval_seed,
            ):
        """ 評価 """
        if eval_seed is not None:
            np_random_state = np.random.get_state()
            random_state = random.getstate()
            np.random.seed(eval_seed)
            random.seed(eval_seed)

        # 乱雑状態のバックアップ
        epsilon_backup = self.agt.epsilon

        # 評価用の乱雑度を設定
        self.agt.epsilon = eval_epsilon

        # 記録クラス初期化
        self.recorder.reset()

        # 学習シミュレーションの準備
        timestep = 1
        episode = 1

        # 学習なしシミュレーション
        obs = self.eval_env.reset()
        while True:
            self.env.pygame_render(f'time:{timestep}/episode:{episode} ' +'.\n' + 'pleas wait. now evaluating')
            
            # エージェントが行動を選ぶ
            act = self.agt.select_action(obs)

            # 環境が報酬と次の観測を決める
            rwd, done, obs = self.eval_env.step(act)

            # 報酬を記録
            self.recorder.add(rwd, done)

            # 終了判定
            timestep += 1
            episode += (done is True)
            if eval_n_step is not None:
                if timestep >= eval_n_step + 1:
                    break
            if eval_n_episode is not None:
                if episode >= eval_n_episode + 1:
                    break

        # 乱雑度の復元
        self.agt.epsilon = epsilon_backup

        # 結果の集計
        mean_rwds, mean_steps = self.recorder.get_result()

        # 乱数状態の復元
        if eval_seed is not None:
            np.random.set_state(np_random_state)
            random.setstate(random_state)

        return mean_rwds, mean_steps

    def _show_Q(self):
        """ Q値の表示 """
        if self.obss is None:
            return

        print('')
        print('学習後のQ値')
        for obs in self.obss:
            q_vals = self.agt.get_Q(np.array(obs))
            if q_vals is not None:
                valstr = [f' {v: .2f}' for v in q_vals]
                valstr = ','.join(valstr)
                print('{}:{}'.format(str(np.array(obs)), valstr))

    def save_history(self, pathname):
        """ 学習履歴を保存 """
        np.savez(
            pathname + '.npz',
            eval_rwds=self.hist_rwds,
            eval_steps=self.hist_steps,
            eval_x=self.hist_time,
            )

    def load_history(self, pathname):
        """ 学習履歴を読み込み """
        hist = np.load(pathname + '.npz')
        self.hist_rwds = hist['eval_rwds'].tolist()
        self.hist_steps = hist['eval_steps'].tolist()
        self.hist_time = hist['eval_x'].tolist()
        self.hist_start_time = self.hist_time[-1]
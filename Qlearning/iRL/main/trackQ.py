import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SAVE_DIR = 'agt_trackQ'

ptype = 'G'  # コマンドライン引数に何も指定しないときのデフォルト値

# コマンドライン引数の処理: 'L'(学習) or 'G'(グラフ表示)
argvs = sys.argv
if len(argvs) == 2:
    ptype = argvs[1]
print('process: ', ptype)

# 調べるエージェント
agt_types = ['netQ', 'replayQ', 'targetQ']

# 保存ファイル名
file_name = SAVE_DIR + '/%s_trackQ'

if ptype == 'L':
    # コマンドライン引数に「L」を指定して、学習させる場合
    from env_corridor import CorridorEnv as Env
    from agt_netQ import NetQAgt
    from agt_replayQ import ReplayQAgt
    from agt_targetQ import TargetQAgt
    from trainer_mod import Trainer

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    for agt_type in agt_types:
        print('\nLearning process: ', agt_type)
        # fname = SAVE_DIR + '/' + agt_type + '_trackQ'
        env = Env()
        eval_env = Env()
        obs = env.reset()

        # エージェント共通パラメータ
        agt_prm = {
            'input_size': obs.shape,
            'n_act': env.n_act,
            'epsilon': 0.5,
            'gamma': 1.0,
        }
        # エージェント個別パラメータを指定して
        # インスタンス生成
        if agt_type == 'netQ':
            agt_prm['n_dense'] = 32
            agt = NetQAgt(**agt_prm)

        elif agt_type == 'replayQ':
            agt_prm['n_dense'] = 32
            agt_prm['replay_memory_size'] = 100
            agt_prm['replay_batch_size'] = 20
            agt = ReplayQAgt(**agt_prm)

        elif agt_type == 'targetQ':
            agt_prm['n_dense'] = 32
            agt_prm['replay_memory_size'] = 100
            agt_prm['replay_batch_size'] = 20
            agt_prm['target_interval'] = 20
            agt = TargetQAgt(**agt_prm)

        # トレーナークラスのインスタンス生成
        trn = Trainer(agt, env, eval_env)
        trn.monitor_x = [0, 1, 2, 0]  # Q値を調べる観測を指定

        # シミュレーション開始
        trn.simulate(
            n_step=15000,  # 15000
            eval_interval=100,
            eval_n_episode=10,
            eval_seed=1,
        )

        # 学習履歴の保存(Q値を含む)
        trn.save_history(file_name % agt_type)

elif ptype == 'G':
    agt_types = ['netQ', 'replayQ', 'targetQ']
    col = ['#15f', '#f59', '#093']
    plt.figure(figsize=(8, 8))
    for j in range(2):
        plt.subplot(2, 1, j + 1)
        for i, agt_type in enumerate(agt_types):
            # fname = SAVE_DIR + '/' + agt_type + '_trackQ.npz'
            dat = np.load(file_name % agt_type + '.npz')
            Ts = dat['eval_x']
            Qs = dat['eval_Qs']
            plt.plot(Ts, Qs[:, 1], '.-', color=col[i],
                     label=agt_type + ' (a=1)')  # 「進む」
        if j == 1:
            plt.ylim(3.6, 4.8)
            plt.xlim(8000, 15000)
        plt.grid(axis='both')
        plt.legend()
    plt.show()

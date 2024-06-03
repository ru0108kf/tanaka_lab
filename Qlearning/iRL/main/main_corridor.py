"""
main_corridor.py
廊下タスクの実行ファイル
"""
import os
import sys

# 自作モジュール
from env_corridor import CorridorEnv
import trainer
import myutil

SAVE_DIR = 'agt_data'
ENV_NAME = 'env_corridor'

argvs = sys.argv
if len(argvs) < 3:
    msg = (
        '\n' +
        '---- 使い方 ---------------------------------------\n' +
        '2つのパラメータを指定して実行します\n\n' +
        '> python main_corridor.py [agt_type] [process_type]\n\n' +
        '[agt_type]\t: tableQ, netQ, replayQ, targetQ\n' +
        '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' +
        '例 > python main_corridor.py tableQ L\n' +
        '---------------------------------------------------'
    )
    print(msg)
    sys.exit()

# 入力パラメータの確認 //////////
agt_type = argvs[1]
process_type = argvs[2]

# 保存用フォルダの確認・作成 //////////
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


# process_type //////////
if process_type in ('learn', 'L'):
    IS_LOAD_DATA = False
    IS_LEARN = True
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
elif process_type in ('more', 'M'):
    IS_LOAD_DATA = True
    IS_LEARN = True
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
elif process_type in ('graph', 'G'):
    IS_LOAD_DATA = False
    IS_LEARN = False
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
    print('[q] 終了')
elif process_type in ('anime', 'A'):
    IS_LOAD_DATA = True
    IS_LEARN = False
    IS_SHOW_GRAPH = False
    IS_SHOW_ANIME = True
    print('[q] 終了')
else:
    print('process type が間違っています。')
    sys.exit()

# Envインスタンス生成 //////////
# 学習用環境
env = CorridorEnv()
obs = env.reset()
# 評価用環境
eval_env = CorridorEnv()

# Trainer シミュレーション共通パラメータ //////////
sim_prm = {
    'n_step': 5000,         # ステップ数
    'n_episode': None,        # エピソードは終了条件にしない
    'is_learn': True,       # 学習する
    'is_eval': True,        # 評価する
    'eval_interval': 100,   # 評価のインターバル
    'eval_n_episode': 10,   # 評価のエピソード数
    'eval_n_step': None,    # 評価でステップ数は終了条件にしない
    'eval_epsilon': 0.0,    # 評価時の乱雑度
    'eval_seed': 1,         # 評価時の乱数のシード値
    'is_animation': False,  # アニメーションの表示はしない
}
obss = [                    # Q値チェック用の観測
        [1, 0, 2, 0],
        [0, 1, 2, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 1],
        [1, 0, 0, 2],
        [0, 1, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ]
sim_prm['obss'] = obss

# Trainer アニメーション共通パラメータ //////////
sim_anime_prm = {
    'n_step': None,           # ステップ数（Noneは終了条件にしない）
    'n_episode': 100,       # エピソード数
    'is_eval': False,       # 評価を行うか
    'is_learn': False,      # 学習を行うか
    'is_animation': True,   # アニメーションの表示をするか
    'anime_delay': 0.2,     # アニメーションのフレーム間の秒数
}
ANIME_EPSILON = 0.0         # アニメーション時の乱雑度

# グラフ表示共通パラメータ //////////
graph_prm = {
    'target_reward': 2.5,   # 報酬の目標値
    'target_step': 3.5,     # ステップ数の目標値
}

# Agt 共通パラメータ //////////
agt_prm = {
    'n_act': env.n_act,
    'epsilon': 0.5,         # 乱雑度
    'gamma': 1.0,           # 割引率
    'filepath': (SAVE_DIR + '/' +
                 ENV_NAME + '_' +
                 agt_type)
}

# agt_type 別のパラメータ //////////
if agt_type == 'tableQ':
    agt_prm['init_val_Q'] = 0            # Q値の初期値
    agt_prm['alpha'] = 0.2               # 学習率

elif agt_type == 'netQ':
    agt_prm['input_size'] = obs.shape    # 入力のサイズ
    agt_prm['n_dense'] = 32              # 中間層のユニット数

elif agt_type == 'replayQ':
    agt_prm['input_size'] = obs.shape    # 入力のサイズ
    agt_prm['n_dense'] = 32              # 中間層のユニット数
    agt_prm['replay_memory_size'] = 100  # 記憶サイズ
    agt_prm['replay_batch_size'] = 20    # バッチサイズ

elif agt_type == 'targetQ':
    agt_prm['input_size'] = obs.shape    # 入力のサイズ
    agt_prm['n_dense'] = 32              # 中間層のユニット数
    agt_prm['replay_memory_size'] = 100  # 記憶サイズ
    agt_prm['replay_batch_size'] = 20    # バッチサイズ
    agt_prm['target_interval'] = 10      # ターゲットインターバル


# メイン //////////
if (IS_LOAD_DATA is True) or \
        (IS_LEARN is True) or \
        (sim_prm['is_animation'] is True):

    # エージェントをインポートしてインスタンス作成
    if agt_type == 'tableQ':
        from agt_tableQ import TableQAgt as Agt
    elif agt_type == 'netQ':
        from agt_netQ import NetQAgt as Agt
    elif agt_type == 'replayQ':
        from agt_replayQ import ReplayQAgt as Agt
    elif agt_type == 'targetQ':
        from agt_targetQ import TargetQAgt as Agt
    else:
        raise ValueError('agt_type が間違っています')

    agt = Agt(**agt_prm)

    # trainer インスタンス作成
    trn = trainer.Trainer(agt, env, eval_env)

    if IS_LOAD_DATA is True:
        # エージェントのデータロード
        try:
            agt.load_weights()
            trn.load_history(agt.filepath)
        except Exception as e:
            print(e)
            print('エージェントのパラメータがロードできません')
            sys.exit()

    if IS_LEARN is True:
        # 学習
        trn.simulate(**sim_prm)
        agt.save_weights()
        trn.save_history(agt.filepath)

    if IS_SHOW_ANIME is True:
        # アニメーション
        agt.epsilon = ANIME_EPSILON
        trn.simulate(**sim_anime_prm)

if IS_SHOW_GRAPH is True:
    # グラフ表示
    myutil.show_graph(agt_prm['filepath'], **graph_prm)

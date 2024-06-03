"""
main_field.py
フィールドタスクの実行ファイル
"""
import os
import sys

# 自作モジュール
from env_field import FieldEnv
from env_field import TaskType
import trainer
import myutil

SAVE_DIR = 'agt_data'
ENV_NAME = 'env_field'

argvs = sys.argv
if len(argvs) < 4:
    msg = (
        '\n' +
        '---- 使い方 ---------------------------------------\n' +
        '3つのパラメータを指定して実行します\n\n' +
        '> python main_field.py [agt_type] [task_type] [process_type]\n\n' +
        '[agt_type]\t: tableQ, netQ, replayQ, targetQ\n' +
        f'[task_type]\t: {", ".join([t.name for t in TaskType])}\n' +
        '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' +
        '例 > python main_field.py tableQ no_wall L\n' +
        '---------------------------------------------------'
    )
    print(msg)
    sys.exit()

# 入力パラメータの確認
agt_type = argvs[1]
task_type = TaskType.Enum_of(argvs[2])
if task_type is None:
    msg = (
        '\n' +
        '[task_type] が異なります。以下から選んで指定してください。\n' +
        f'{", ".join([t.name for t in TaskType])}\n'
    )
    print(msg)
    sys.exit()
process_type = argvs[3]

# 保存用フォルダの確認・作成
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# process_typeでパラメータをセット (A)
if process_type in ('learn', 'L'):
    IS_LOAD_DATA = False    # 学習したデータを読み込むかどうか
    IS_LEARN = True         # 学習を行うかどうか
    IS_SHOW_GRAPH = True    # 学習曲線のグラフを表示するかどうか
    IS_SHOW_ANIME = False   # 画像によるシミュレーションの表示をするかどうか
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

# 環境の設定 /////////////////////////////// (A)
# Envインスタンス生成
# 学習用環境
env = FieldEnv()
env.set_task_type(task_type)
obs = env.reset()  # obsのサイズがエージェントのパラメータ設定で使われる

# 評価用環境
eval_env = FieldEnv()
eval_env.set_task_type(task_type)

# エージェントのパラメータ設定 /////////////// (B)
# Agt 共通パラメータ
agt_prm = {
    'n_act': env.n_act,
    'filepath': (SAVE_DIR + '/' +
                 ENV_NAME + '_' +
                 agt_type + '_' +
                 task_type.name)
}
# agt_type 別のパラメータ
if agt_type == 'tableQ':
    agt_prm['init_val_Q'] = 0            # Q値の初期値
    agt_prm['alpha'] = 0.1               # 学習率

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


# トレーナーのパラメータ設定 ////////////////
# シミュレーション共通パラメータ (C)
sim_prm = {
    'n_step': 1000,         # ステップ数
    'n_episode': None,      # エピソードは終了条件にしない
    'is_learn': True,       # 学習する
    'is_eval': True,        # 評価する
    'eval_interval': 100,   # 評価のインターバル
    'eval_n_episode': 10,   # 評価のエピソード数
    'eval_n_step': None,    # 評価でステップ数は終了条件にしない
    'eval_epsilon': 0.0,    # 評価時の乱雑度
    'eval_seed': 1,         # 評価時の乱数のシード値
    'is_animation': False,  # アニメーションの表示はしない
}

# アニメーション共通パラメータ (D)
sim_anime_prm = {
    'n_step': None,         # ステップ数は終了条件にしない
    'n_episode': 100,       # エピソード数
    'seed': 1,              # 乱数のシード値を指定
    'is_eval': False,       # 評価しない
    'is_learn': False,      # 学習しない
    'is_animation': True,   # アニメーションを表示する
    'anime_delay': 0.2,     # フレーム時間(秒)
}
ANIME_EPSILON = 0.0

# task_type 別のパラメータ ///////////////// (E)
graph_prm = {}
if task_type == TaskType.no_wall:
    sim_prm['n_step'] = 5000            # ステップ数
    sim_prm['eval_interval'] = 200      # 評価を何ステップごとにするか
    sim_prm['eval_n_episode'] = 100     # 評価のエピソード数
    agt_prm['epsilon'] = 0.2            # 乱雑度
    agt_prm['gamma'] = 0.9              # 割引率
    graph_prm['target_reward'] = 0.71   # グラフの赤い点線の値
    graph_prm['target_step'] = 3.87     # グラフの赤い点線の値

elif task_type == TaskType.fixed_wall:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 200
    sim_prm['eval_n_episode'] = 1
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 0.9
    graph_prm['target_reward'] = 1.0
    graph_prm['target_step'] = 12.0

elif task_type == TaskType.random_wall:
    sim_prm['n_step'] = 50000
    sim_prm['eval_interval'] = 1000
    sim_prm['eval_n_episode'] = 100
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 0.9
    graph_prm['target_reward'] = 1.6
    graph_prm['target_step'] = 22.0
    # 数値を設定すると
    # rewards/episodeがこの値を超えた時に
    # 学習シミュレーションが終了する
    sim_prm['eary_stop'] = 1.6

"""  # ---- my_wall追加時にはこのコメント行を消す
elif task_type == TaskType.my_wall:  # <--------------- 追加 ここから
    # 学習シミュレーションのパラメータ
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 1000
    sim_prm['eval_n_episode'] = 10
    sim_prm['eval_seed'] = 1

    # エージェントのパラメータ
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 0.9

    # グラフに目標値を入れる場合に数値を設定
    graph_prm['target_reward'] = None
    graph_prm['target_step'] = None  # <-------------- 追加 ここまで
"""  # ---- my_wall追加時にはこのコメント行を消す

# メイン ///////////////////////////////// (F)
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

    # エージェントのインスタンス生成(G)
    agt = Agt(**agt_prm)

    # trainer インスタンス作成(H)
    trn = trainer.Trainer(agt, env, eval_env)

    if IS_LOAD_DATA is True:
        # エージェントのデータロード(I)
        try:
            agt.load_weights()
            trn.load_history(agt.filepath)
        except Exception as e:
            print(e)
            print('エージェントのパラメータがロードできません')
            sys.exit()

    if IS_LEARN is True:
        # 学習(J)
        trn.simulate(**sim_prm)
        agt.save_weights()
        trn.save_history(agt.filepath)

    if IS_SHOW_ANIME is True:
        # アニメーション(K)
        agt.epsilon = ANIME_EPSILON
        trn.simulate(**sim_anime_prm)

if IS_SHOW_GRAPH is True:
    # グラフ表示(L)
    myutil.show_graph(agt_prm['filepath'], **graph_prm)

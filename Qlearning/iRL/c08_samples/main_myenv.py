"""
main_myenv.py
"""
import myutil
from env_myenv import MyEnv
from agt_tableQ import TableQAgt as Agt  # <--- コメントアウト (A)
# from agt_myagt import MyAgt as Agt  # <----- 追加 (B)
from trainer import Trainer


env = MyEnv()
eval_env = MyEnv()
obs = env.reset()

agt = Agt(
    n_act=env.n_act,  # <----コメントアウト (C)
    alpha=0.01, epsilon=0.5,
    )

trn = Trainer(agt, env, eval_env)

print('学習前動作確認')
trn.simulate(
    is_animation=True, anime_delay=0.1,
    is_learn=False, n_episode=20)

print('学習中')
trn.simulate(
    n_step=5000,
    eval_interval=1000,
    eval_n_episode=100,
    eval_seed=1,
    obss=[[0]])

trn.save_history('tmp')
myutil.show_graph('tmp')

print('学習後動作確認')
agt.epsilon = 0
trn.simulate(
    is_animation=True, anime_delay=0.1,
    is_learn=False, n_episode=20)

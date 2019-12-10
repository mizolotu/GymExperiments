import os, shutil
import os.path as osp
from datetime import datetime

from baselines.common.vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn as learn_ppo
from baselines.acktr.acktr import learn as learn_acktr
from baselines.ddpg.ddpg import learn as learn_ddpg
from baselines import logger
from baselines.common.tf_util import launch_tensorboard_in_background

from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.bipedal_walker import BipedalWalker
from gym.envs.box2d.lunar_lander import LunarLanderContinuous

def clean_dir(dir_name):
    for the_file in os.listdir(dir_name):
        file_path = osp.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def test_alg_on_env(env_class, algorithm, network, ne, ns, tt, ld='log'):
    env_fns = [env_class for _ in range(ne)]
    train_envs = SubprocVecEnv(env_fns)
    eval_envs = SubprocVecEnv(env_fns)
    logdir = '{0}/{1}/{2}_{3}/'.format(ld, env_class.__name__, algorithm['name'], network)
    tb_dir = logdir + '/tb'
    if osp.isdir(logdir): clean_dir(tb_dir)
    format_strs = os.getenv('', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)
    now = datetime.now()
    tag = now.strftime('%m/%d/%Y_%H:%M:%S')
    launch_tensorboard_in_background(tb_dir)
    algorithm['learn'](env=train_envs, network=network, nsteps=ns, total_timesteps=tt, log_interval=int(tt/(ne*ns*100)))

if __name__ == '__main__':

    env_classes = [
        PendulumEnv,
        Continuous_MountainCarEnv,
        CarRacing,
        LunarLanderContinuous,
        BipedalWalker
    ]

    algorithms = [
        {'name': 'ppo', 'learn': learn_ppo},
        {'name': 'acktr', 'learn': learn_acktr},
        {'name': 'ddpg', 'learn': learn_ddpg}
    ]

    networks = [
        'mlp',
        'cnn',
        'lstm',
        'resnet-mlp',
        'resnet-cnn',
        'attention-cnn'
    ]

    n_envs = 16
    n_steps = 125
    n_episodes = 5000
    total_timesteps = n_episodes * n_steps * n_envs
    print('Total time steps: {0}'.format(total_timesteps))

    test_alg_on_env(env_classes[0], algorithms[0], networks[0], ne=n_envs, ns=n_steps, tt=total_timesteps)


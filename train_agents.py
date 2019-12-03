from baselines.common.vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn as learn_ppo
from baselines.acktr.acktr import learn as learn_acktr
from baselines.ddpg.ddpg import learn as learn_ddpg
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.bipedal_walker import BipedalWalker
from gym.envs.box2d.lunar_lander import LunarLanderContinuous

def test_alg_on_env(env_class, algorithm, network, ne, tt):
    env_fns = [env_class for _ in range(ne)]
    train_env = SubprocVecEnv(env_fns)
    algorithm['learn'](env=train_env, network=network, total_timesteps=tt)

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
        'resnet_mlp',
        'impala_cnn',
        'att_cnn'
    ]

    n_envs = 5
    total_timesteps = 100 * 2000 * 5

    test_alg_on_env(env_classes[0], algorithms[0], networks[0], ne=n_envs, tt=total_timesteps)


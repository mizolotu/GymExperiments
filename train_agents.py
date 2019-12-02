from baselines.common.vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn as learn_ppo
from baselines.acktr.acktr import learn as learn_acktr
from baselines.ddpg.ddpg import learn as learn_ddpg
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.bipedal_walker import BipedalWalker
from gym.envs.box2d.lunar_lander import LunarLanderContinuous

if __name__ == '__main__':
    env_classes = [
        PendulumEnv,
        Continuous_MountainCarEnv,
        CarRacing,
        LunarLanderContinuous,
        BipedalWalker
    ]
    n_envs = 4
    env_fns = [env_classes[0] for _ in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)
    eval_env = SubprocVecEnv(env_fns)
    learn_ddpg(env=train_env, network='lstm', total_timesteps=100000, eval_env=eval_env)
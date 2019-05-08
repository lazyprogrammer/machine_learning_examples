# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import argparse
import os
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind, Monitor
from a2c import Agent
from neural_network import CNN
import imageio
import time


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    return parser.parse_args()


def get_agent(env, nsteps=5, nstack=1, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              epsilon=1e-5, alpha=0.99):
    # Note: nstack=1 since frame_stack=True, during training frame_stack=False
    agent = Agent(Network=CNN, ob_space=env.observation_space,
                  ac_space=env.action_space, nenvs=1, nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    return agent


def main():
    env_id = get_args().env
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=True)
    env = Monitor(env)
    # rewards will appear higher than during training since rewards are not clipped

    agent = get_agent(env)

    # check for save path
    save_path = os.path.join('models', env_id + '.save')
    agent.load(save_path)

    obs = env.reset()
    renders = []
    while True:
        obs = np.expand_dims(obs.__array__(), axis=0)
        a, v = agent.step(obs)
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            print(info)
            env.reset()


if __name__ == '__main__':
    main()

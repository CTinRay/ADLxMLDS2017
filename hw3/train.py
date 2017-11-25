import argparse
import pdb
import sys
import traceback
import gym
from atari_wrappers import wrap_deepmind
from agent_dqn import AgentDQN


def main(args):
    env = gym.make('Breakout-v0')
    env = wrap_deepmind(env, frame_stack=True)
    train_args = {
        'max_timesteps': 1000000,
        'gamma': 0.9,
        'buffer_size': 10000,
        'exploration_final_eps': 0.05,
        'batch_size': 16,
        'prioritized_replay_eps': 1e-4,
        'target_network_update_freq': 100
    }
    agent_dqn = AgentDQN(env, train_args)
    agent_dqn.train()


def parse_arg():
    parser = argparse.ArgumentParser(description='ADL HW3')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    try:
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

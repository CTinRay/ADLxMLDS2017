import argparse
import pdb
import sys
import traceback
import gym
from atari_wrappers import wrap_deepmind
from agent_dqn import AgentDQN


def main(args):
    env = gym.make('Breakout-v0')
    env = wrap_deepmind(env, clip_rewards=False, episode_life=False,
                        frame_stack=True)
    train_args = {
        'max_timesteps': 1000000,
        'gamma': 0.99999,
        'buffer_size': 100000,
        'exploration_final_eps': 0.05,
        'batch_size': 16,
        'prioritized_replay_eps': 1e-4,
        'target_network_update_freq': 100,
        'learning_rate': 1e-3
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

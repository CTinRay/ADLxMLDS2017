import pdb
import argparse
from matplotlib import pyplot as plt
import numpy as np


def read_file(filename):
    timesteps = []
    rewards = []
    with open(args.log_file) as f:
        for l in f:
            cols = l.split(',')
            timesteps.append(float(cols[0]))
            rewards.append(float(cols[1]))

    return timesteps, rewards


def main(args):
    timesteps, rewards = read_file(args.log_file)
    timesteps = np.array(timesteps)
    rewards = np.array(rewards)
    rewards = np.convolve(rewards,
                          np.ones((args.window_size,)) / args.window_size,
                          mode='same')
    fig = plt.figure(dpi=450)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Reward over time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Avg. Reward')

    ax.plot(timesteps, rewards)
    fig.savefig(args.output)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot learning curve.')
    parser.add_argument('log_file', type=str,
                        help='Path to the log file.')
    parser.add_argument('output', type=str,
                        help='Directory of the data.')
    parser.add_argument('--window_size', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

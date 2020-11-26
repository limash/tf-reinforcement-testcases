import logging

import numpy as np
import matplotlib.pyplot as plt
import gym

from tf_reinforcement_testcases import deep_q_learning


if __name__ == '__main__':
    cart_pole = 'CartPole-v1'
    halite = 'gym_halite:halite-v0'

    agent = deep_q_learning.DQNAgent(cart_pole)

    # set to logging.WARNING to disable logs or logging.DEBUG to see losses as well
    # logging.getLogger().setLevel(logging.INFO)
    model = agent.train()
    # print("Finished training! Testing...")
    # print(f"Total Episode Reward is {agent.test(env)}")

    # plt.style.use('seaborn')
    # plt.plot(np.arange(0, len(rewards_history), 5), rewards_history[::5])
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.show()

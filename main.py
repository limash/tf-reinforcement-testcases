import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tf messages

import pickle

import ray
import numpy as np

from tf_reinforcement_testcases import deep_q_learning, storage, misc

AGENTS = {"regular": deep_q_learning.RegularDQNAgent,
          "fixed": deep_q_learning.FixedQValuesDQNAgent,
          "double": deep_q_learning.DoubleDQNAgent,
          "double_dueling": deep_q_learning.DoubleDuelingDQNAgent,
          "priority_dd": deep_q_learning.PriorityDoubleDuelingDQNAgent}


def one_call(env_name, agent_object, data):
    batch_size = 64
    n_steps = 2
    buffer = storage.UniformBuffer(min_size=batch_size)

    agent = agent_object(env_name,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         n_steps,
                         data)
    weights, mask, reward = agent.train(iterations_number=2000)

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Done")


def multi_call(env_name, agent_object, data):
    ray.init()
    parallel_calls = 8
    batch_size = 64
    n_steps = 2
    buffer = storage.UniformBuffer(min_size=batch_size)

    agents = [agent_object.remote(env_name,
                                  buffer.table_name, buffer.server_port, buffer.min_size,
                                  n_steps,
                                  data) for _ in range(parallel_calls)]
    futures = [agent.train.remote(iterations_number=2000) for agent in agents]
    outputs = ray.get(futures)

    if data is None:
        rewards = np.empty(parallel_calls)
        weights_list, mask_list = [], []
        for count, (weights, mask, reward) in enumerate(outputs):
            weights_list.append(weights)
            mask_list.append(mask)
            rewards[count] = reward
            misc.plot_2d_array(weights[0], "Zero_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
            misc.plot_2d_array(weights[2], "First_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
        argmax = rewards.argmax()
        data = {
            'weights': weights_list[argmax],
            'mask': mask_list[argmax],
            'reward': rewards[argmax]
        }
        with open('data/data.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    cart_pole = 'CartPole-v1'
    halite = 'gym_halite:halite-v0'

    try:
        with open('data/data.pickle', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = None

    multi_call(cart_pole, AGENTS['regular'], data)

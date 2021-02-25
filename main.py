# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tf messages

import pickle

import ray
import numpy as np
import tensorflow as tf

from tf_reinforcement_testcases import deep_q_learning, actor_critic, storage, misc

AGENTS = {"regular": deep_q_learning.RegularDQNAgent,
          "fixed": deep_q_learning.FixedQValuesDQNAgent,
          "double": deep_q_learning.DoubleDQNAgent,
          "double_dueling": deep_q_learning.DoubleDuelingDQNAgent,
          "categorical": deep_q_learning.CategoricalDQNAgent,
          "priority_categorical": deep_q_learning.PriorityCategoricalDQNAgent,
          "actor_critic": actor_critic.ACAgent}

BUFFERS = {"regular": storage.UniformBuffer,
           "fixed": storage.UniformBuffer,
           "double": storage.UniformBuffer,
           "double_dueling": storage.UniformBuffer,
           "categorical": storage.UniformBuffer,
           "priority_categorical": storage.PriorityBuffer,
           "actor_critic": storage.UniformBuffer}


def one_call(env_name, agent_name, data, make_sparse):

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    batch_size = 64
    n_steps = 2
    init_sample_eps = 1.  # 1 means random sampling
    eps = .5  # start for polynomial decay eps schedule, it should be real (double)

    buffer = BUFFERS[agent_name](min_size=batch_size)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         n_steps,
                         data, make_sparse,
                         init_epsilon=init_sample_eps)
    weights, mask, reward = agent.train(iterations_number=10000, epsilon=eps)

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Done")


def multi_call(env_name, agent_name, data, make_sparse, plot=False):

    parallel_calls = 2

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024) for _ in range(parallel_calls)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    ray.init(num_cpus=parallel_calls, num_gpus=1)

    batch_size = 64
    n_steps = 2
    init_sample_eps = 1.  # 1 means random sampling
    eps = .5  # start for polynomial decay eps schedule, it should be real (double)

    buffer = BUFFERS[agent_name](min_size=batch_size)

    agent_object = AGENTS[agent_name]
    agent_object = ray.remote(num_gpus=1/parallel_calls)(agent_object)
    agents = [agent_object.remote(env_name,
                                  buffer.table_name, buffer.server_port, buffer.min_size,
                                  n_steps, data, make_sparse,
                                  init_epsilon=init_sample_eps) for _ in range(parallel_calls)]
    futures = [agent.train.remote(iterations_number=10000, epsilon=eps) for agent in agents]
    outputs = ray.get(futures)

    rewards = np.empty(parallel_calls)
    weights_list, mask_list = [], []
    for count, (weights, mask, reward) in enumerate(outputs):
        weights_list.append(weights)
        mask_list.append(mask)
        rewards[count] = reward
        print(f"Proc #{count}: reward = {reward}")
        if plot:
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
    goose = 'gym_goose:goose-v0'
    breakout = 'BreakoutNoFrameskip-v4'

    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    one_call(breakout, 'double', init_data, make_sparse=False)

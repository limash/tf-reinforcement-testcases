# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tf messages

import pickle

import ray
import numpy as np

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

BATCH_SIZE = 64
BUFFER_SIZE = 100000
N_STEPS = 2  # 2 steps is a regular TD(0)

INIT_SAMPLE_EPS = .1  # 1 means random sampling, for sampling before training
INIT_N_SAMPLES = 100000

EPS = .1  # start for polynomial decay eps schedule, it should be real (double)


def one_call(env_name, agent_name, data, make_sparse):

    buffer = BUFFERS[agent_name](min_size=BATCH_SIZE, max_size=BUFFER_SIZE)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name, INIT_N_SAMPLES,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         N_STEPS, INIT_SAMPLE_EPS,
                         data, make_sparse)
    weights, mask, reward = agent.train_collect(iterations_number=10000, epsilon=EPS)

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Done")


def multi_call(env_name, agent_name, data, make_sparse, plot=False):

    parallel_calls = 4
    ray.init(num_cpus=parallel_calls, num_gpus=1)

    buffer = BUFFERS[agent_name](min_size=BATCH_SIZE)

    agent_object = AGENTS[agent_name]
    agent_object = ray.remote(num_gpus=1/parallel_calls)(agent_object)
    agents = [agent_object.remote(env_name, INIT_N_SAMPLES,
                                  buffer.table_name, buffer.server_port, buffer.min_size,
                                  N_STEPS, INIT_SAMPLE_EPS,
                                  data, make_sparse) for _ in range(parallel_calls)]
    futures = [agent.train_collect.remote(iterations_number=60000, epsilon=EPS) for agent in agents]
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

    one_call(breakout, 'regular', init_data, make_sparse=False)

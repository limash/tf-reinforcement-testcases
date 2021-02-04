import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tf messages

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


def one_call(env_name, agent_name, data, make_sparse):
    batch_size = 64
    n_steps = 2
    buffer = BUFFERS[agent_name](min_size=batch_size)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         n_steps,
                         data, make_sparse)
    weights, mask, reward = agent.train(iterations_number=2000)

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Done")


def multi_call(env_name, agent_name, data, make_sparse, plot=False):
    ray.init()
    parallel_calls = 10
    batch_size = 64
    n_steps = 2
    buffer = BUFFERS[agent_name](min_size=batch_size)

    agent_object = AGENTS[agent_name]
    agent_object = ray.remote(agent_object)
    agents = [agent_object.remote(env_name,
                                  buffer.table_name, buffer.server_port, buffer.min_size,
                                  n_steps,
                                  data, make_sparse) for _ in range(parallel_calls)]
    futures = [agent.train.remote(iterations_number=2000) for agent in agents]
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

    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    multi_call(goose, 'categorical', init_data, make_sparse=False)

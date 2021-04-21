# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tf messages

import pickle
from pathlib import Path

import ray
import reverb
import numpy as np

from tf_reinforcement_testcases import deep_q_learning, policy_gradient, storage, misc

from config import *

config = CONF_DQN

AGENTS = {"regular": deep_q_learning.RegularDQNAgent,
          "fixed": deep_q_learning.FixedQValuesDQNAgent,
          "double": deep_q_learning.DoubleDQNAgent,
          "double_dueling": deep_q_learning.DoubleDuelingDQNAgent,
          "categorical": deep_q_learning.CategoricalDQNAgent,
          "priority_categorical": deep_q_learning.PriorityCategoricalDQNAgent,
          "actor_critic": policy_gradient.ACAgent}

BUFFERS = {"regular": storage.UniformBuffer,
           "fixed": storage.UniformBuffer,
           "double": storage.UniformBuffer,
           "double_dueling": storage.UniformBuffer,
           "categorical": storage.UniformBuffer,
           "priority_categorical": storage.PriorityBuffer,
           "actor_critic": storage.UniformBuffer}

BATCH_SIZE = config["batch_size"]
BUFFER_SIZE = config["buffer_size"]
INIT_N_SAMPLES = config["init_n_samples"]


def one_call(env_name, agent_name, data, checkpoint, make_sparse):
    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None
    buffer = BUFFERS[agent_name](min_size=BATCH_SIZE, max_size=BUFFER_SIZE, checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name, INIT_N_SAMPLES,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         config,
                         data, make_sparse, make_checkpoint=True)
    weights, mask, reward, checkpoint = agent.train_collect()

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)
    print("Done")


def multi_call(env_name, agent_name, data, checkpoint, make_sparse, plot=False):
    parallel_calls = 3
    ray.init(num_cpus=parallel_calls, num_gpus=1)

    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None
    buffer = BUFFERS[agent_name](min_size=BATCH_SIZE, max_size=BUFFER_SIZE, checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]
    agent_object = ray.remote(num_gpus=1 / parallel_calls)(agent_object)
    agents = []
    for i in range(parallel_calls):
        make_checkpoint = True if i == 0 else False  # make a checkpoint only in the first worker
        agents.append(agent_object.remote(env_name, INIT_N_SAMPLES,
                                          buffer.table_name, buffer.server_port, buffer.min_size,
                                          N_STEPS, INIT_SAMPLE_EPS,
                                          data, make_sparse, make_checkpoint))
    futures = [agent.train_collect.remote(iterations_number=20000, epsilon=EPS) for agent in agents]
    outputs = ray.get(futures)

    rewards = np.empty(parallel_calls)
    weights_list, mask_list = [], []
    for count, (weights, mask, reward, _) in enumerate(outputs):
        weights_list.append(weights)
        mask_list.append(mask)
        rewards[count] = reward
        print(f"Proc #{count}: Average reward = {reward:.2f}")
        if plot:
            misc.plot_2d_array(weights[0], "Zero_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
            misc.plot_2d_array(weights[2], "First_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
    argmax = rewards.argmax()
    print(f"Reward to save: {rewards[argmax]:.2f}")
    data = {
        'weights': weights_list[argmax],
        'mask': mask_list[argmax],
        'reward': rewards[argmax]
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    _, _, _, checkpoint = outputs[0]
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    # cart_pole = 'CartPole-v1'
    # goose = 'gym_goose:goose-v0'
    # breakout = 'BreakoutNoFrameskip-v4'

    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    try:
        init_checkpoint = open('data/checkpoint', 'r').read()
    except FileNotFoundError:
        init_checkpoint = None

    one_call(config["environment"], config["agent"], init_data, init_checkpoint, make_sparse=False)

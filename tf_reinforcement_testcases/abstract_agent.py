import abc
# import time
# import copy
import itertools as it

import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym
import reverb

from tf_reinforcement_testcases import models, storage


class Agent(abc.ABC):
    NETWORKS = {'CartPole-v1': models.get_q_mlp,
                'CartPole-v1_duel': models.get_dueling_q_mlp}

    def __init__(self, env_name,
                 buffer_table_name, buffer_server_port, buffer_min_size,
                 n_steps=2,
                 data=None):
        # environments; their hyperparameters
        self._train_env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        self._n_outputs = self._train_env.action_space.n  # number of actions
        self._input_shape = self._train_env.observation_space.shape

        # it contains weighs, masks, and a corresponding reward
        self._data = data

        # networks
        self._model = None
        self._target_model = None

        # fraction of random exp sampling
        self._epsilon = 0.1

        # hyperparameters for optimization
        self._optimizer = keras.optimizers.Adam(lr=1e-3)
        self._loss_fn = keras.losses.mean_squared_error

        # buffer; hyperparameters for a reward calculation
        self._table_name = buffer_table_name
        # an object with a client, which is used to store data on a server
        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')
        # make a batch size equal of a minimal size of a buffer
        self._sample_batch_size = buffer_min_size
        self._n_steps = n_steps  # 1. amount of steps stored per item, it should be at least 2;
        # 2. for details see function _collect_trajectories_from_episode()
        # initialize a dataset to be used to sample data from a server
        self._dataset = storage.initialize_dataset(buffer_server_port, buffer_table_name,
                                                   self._input_shape, self._sample_batch_size, self._n_steps)
        self._iterator = iter(self._dataset)
        self._discount_rate = tf.constant(0.95, dtype=tf.float32)
        self._items_created = 0
        self._items_sampled = 0

        # parameters for prioritized exp replay
        self._beta = None
        self._beta_increment = None

    @tf.function
    def _predict(self, observation):
        return self._model(observation)

    def _epsilon_greedy_policy(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self._n_outputs)
        else:
            obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs)
            # Q_values = self._model(obs)
            Q_values = self._predict(obs)
            return np.argmax(Q_values[0])

    def _evaluate_episode(self, epsilon=0):
        """
        epsilon 0 corresponds to greedy policy
        """
        obs = self._eval_env.reset()
        rewards = 0
        while True:
            # if epsilon=0, greedy is disabled
            action = self._epsilon_greedy_policy(obs, epsilon)
            obs, reward, done, info = self._eval_env.step(action)
            rewards += reward
            if done:
                break
        return rewards

    def _evaluate_episodes_greedy(self, num_episodes=3):
        episode_rewards = 0
        for _ in range(num_episodes):
            episode_rewards += self._evaluate_episode()
        return episode_rewards / num_episodes

    def _collect_trajectories_from_episode(self, epsilon):
        """
        Collects trajectories (items) to a buffer.
        A buffer contains items, each item consists of n_steps 'time steps';
        for a regular TD(0) update an item should have 2 time steps.
        One 'time step' contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to obs, reward prior to obs, if is it done at the current obs.
        """
        start_itemizing = self._n_steps - 2
        with self._replay_memory_client.writer(max_sequence_length=self._n_steps) as writer:
            obs = self._train_env.reset()
            action, reward, done = tf.constant(-1), tf.constant(0.), tf.constant(0.)
            obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), obs)
            writer.append((action, obs, reward, done))
            for step in it.count(0):
                action = self._epsilon_greedy_policy(obs, epsilon)
                obs, reward, done, info = self._train_env.step(action)
                action = tf.convert_to_tensor(action, dtype=tf.int32)
                reward = tf.convert_to_tensor(reward, dtype=tf.float32)
                done = tf.convert_to_tensor(done, dtype=tf.float32)
                obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), obs)
                writer.append((action, obs, reward, done))
                if step >= start_itemizing:
                    writer.create_item(table=self._table_name, num_timesteps=self._n_steps, priority=1.)
                    self._items_created += 1
                if done:
                    break

    def _collect_several_episodes(self, epsilon, n_episodes):
        for i in range(n_episodes):
            self._collect_trajectories_from_episode(epsilon)
            # print(f"Collected {i} episodes")

    def _prepare_td_arguments(self, actions, observations, rewards, dones):
        exponents = tf.expand_dims(tf.range(self._n_steps - 1, dtype=tf.float32), axis=1)
        gammas = tf.fill([self._n_steps - 1, 1], self._discount_rate.numpy())
        discounted_gammas = tf.pow(gammas, exponents)

        total_rewards = tf.squeeze(tf.matmul(rewards[:, 1:], discounted_gammas))
        first_observations = tf.nest.map_structure(lambda x: x[:, 0, ...], observations)
        last_observations = tf.nest.map_structure(lambda x: x[:, -1, ...], observations)
        last_dones = dones[:, -1]
        last_discounted_gamma = self._discount_rate ** (self._n_steps - 1)
        second_actions = actions[:, 1]
        return total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions

    @abc.abstractmethod
    def _training_step(self, actions, observations, rewards, dones, info):
        raise NotImplementedError

    def train(self, iterations_number=10000):
        # best_score = 0

        step_counter = 0
        eval_interval = 100
        target_model_update_interval = 100

        weights = None
        mask = None
        mean_episode_reward = 0

        # weights = self._model.get_weights()
        # old_weights = copy.deepcopy(weights)

        for iteration in range(iterations_number):
            # collecting
            # t0 = time.time()
            # do not collect new experience if we have not used previous
            items_created = self._replay_memory_client.server_info()[self._table_name][5].insert_stats.completed
            # if self._items_created < self._items_sampled:
            if items_created < self._items_sampled:
                self._collect_trajectories_from_episode(self._epsilon)
            # t1 = time.time()

            # dm-reverb returns tensors
            sample = next(self._iterator)
            action, obs, reward, done = sample.data
            key, probability, table_size, priority = sample.info
            experiences, info = (action, obs, reward, done), (key, probability, table_size, priority)

            self._items_sampled += self._sample_batch_size

            # training
            # t2 = time.time()
            self._training_step(*experiences, info=info)
            step_counter += 1
            # t3 = time.time()
            # if done:
            #     obs = self._train_env.reset()

            if step_counter % eval_interval == 0:
                mean_episode_reward = self._evaluate_episodes_greedy()
                print("\rTraining step: {}, reward: {}, eps: {:.3f}".format(step_counter,
                                                                            mean_episode_reward,
                                                                            self._epsilon))
                # print(f"Created items count: {self._items_created}")
                print(f"Created items count: {items_created}")
                print(f"Sampled items count: {self._items_sampled}")
            #     print(f"Time spend for sampling is {t1 - t0}")
            #     print(f"Time spend for training is {t3 - t2}")

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # make a sparse model at the last step
            if step_counter % iterations_number == 0:
                # if there is no data available, make a sparse net
                if self._data is None:
                    weights = self._model.get_weights()
                    mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))
                # write the old data else
                else:
                    weights = self._data['weights']
                    mask = self._data['mask']
                    mean_episode_reward = self._data['reward']

        # self._model.set_weights(best_weights)
        return weights, mask, mean_episode_reward

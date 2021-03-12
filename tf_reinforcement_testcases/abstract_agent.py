import abc
# import time
import itertools as it

import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.utils import losses_utils
import gym
import gym.wrappers as gw
import reverb

from tf_reinforcement_testcases import storage, models


class Agent(abc.ABC):
    NETWORKS = {'CartPole-v1': models.get_mlp,
                'BreakoutNoFrameskip-v4': models.get_conv_channels_first
                }
    OBS_DTYPES = {'CartPole-v1': tf.float32,
                  'BreakoutNoFrameskip-v4': tf.uint8
                  }

    def __init__(self, env_name,
                 buffer_table_name, buffer_server_port, buffer_min_size,
                 n_steps, init_epsilon,
                 data=None, make_sparse=False, make_checkpoint=False,
                 ):
        # environments; their hyperparameters
        self._env_name = env_name
        self._train_env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        if env_name == 'BreakoutNoFrameskip-v4':
            self._train_env = gw.FrameStack(
                gw.TimeLimit(
                    gw.AtariPreprocessing(self._train_env),  # includes frameskip 4
                    max_episode_steps=10000),
                4)
            self._eval_env = gw.FrameStack(
                gw.TimeLimit(
                    gw.AtariPreprocessing(self._eval_env),
                    max_episode_steps=10000),
                4)
            self._collect_trajectories_from_episode = self._collect_trajectories_from_episode_atari
        else:
            self._collect_trajectories_from_episode = self._collect_trajectories_from_episode_regular
        self._n_outputs = self._train_env.action_space.n  # number of actions
        self._input_shape = self._train_env.observation_space.shape

        # data contains weighs, masks, and a corresponding reward
        self._data = data
        self._is_sparse = make_sparse
        assert not (not data and make_sparse), "Making a sparse model needs data of weights and mask"
        self._make_checkpoint = make_checkpoint

        # networks
        self._model = None
        self._target_model = None

        # fraction of random exp sampling
        self._epsilon = init_epsilon
        self._repeat_limit = 100  # if there is more similar actions, reset environment

        # hyperparameters for optimization
        # self._optimizer = tf.keras.optimizers.Adam(lr=1.e-5)
        self._optimizer = tf.keras.optimizers.Adam(lr=1.e-4)
        # self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        # self._optimizer = tf.keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
        #                                               epsilon=0.00001, centered=True)
        # self._loss_fn = tf.keras.losses.mean_squared_error
        # self._loss_fn = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.NONE)
        # self._loss_fn = tf.keras.losses.Huber(reduction=losses_utils.ReductionV2.NONE)
        self._loss_fn = tf.keras.losses.Huber()

        # buffer; hyperparameters for a reward calculation
        self._table_name = buffer_table_name
        # an object with a client, which is used to store data on a server
        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')
        # make a batch size equal of a minimal size of a buffer
        self._sample_batch_size = buffer_min_size
        # N_steps - amount of steps stored per item, it should be at least 2;
        # for details see function _collect_trajectories_from_episode()
        self._n_steps = n_steps
        # initialize a dataset to be used to sample data from a server
        # todo: it takes a lot of memory, so it can be useful to separate samplers from trainers in some cases
        self._dataset = storage.initialize_dataset(buffer_server_port, buffer_table_name,
                                                   self._input_shape, self.OBS_DTYPES[env_name],
                                                   self._sample_batch_size, self._n_steps)
        self._iterator = iter(self._dataset)
        self._discount_rate = tf.constant(0.99, dtype=tf.float32)
        self._items_sampled = 0

    @tf.function
    def _predict(self, observation):
        return self._model(observation)

    def _epsilon_greedy_policy(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self._n_outputs)
        else:
            obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs)
            Q_values = self._predict(obs)
            return np.argmax(Q_values[0])

    def _evaluate_episode(self, epsilon=0):
        """
        epsilon 0 corresponds to greedy policy
        """
        obs = self._eval_env.reset()
        rewards = 0

        repeat_counter = 0
        prev_action = -1

        for step in it.count(0):
            action = self._epsilon_greedy_policy(obs, epsilon)

            if action == prev_action:
                repeat_counter += 1
            else:
                repeat_counter = 0
            prev_action = action

            obs, reward, done, info = self._eval_env.step(action)
            rewards += reward
            if done:
                break
            if repeat_counter > self._repeat_limit:
                break
        return rewards, step

    def _evaluate_episodes_greedy(self, num_episodes=3):
        episode_rewards = 0
        for i in range(num_episodes):
            # t1 = time.time()
            rewards, steps = self._evaluate_episode()
            # t2 = time.time()
            # print(f"Evaluation. Episode: {i}; Episode reward: {rewards}; Steps: {steps}; Time: {t2-t1}")
            episode_rewards += rewards
        return episode_rewards / num_episodes

    def _collect_trajectories_from_episode_regular(self, epsilon):
        """
        Collects trajectories (items) to a buffer.
        A buffer contains items, each item consists of n_steps 'time steps';
        for a regular TD(0) update an item should have 2 time steps.
        One 'time step' contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to the obs, reward prior the obs, if is it done at the current obs.
        """
        start_itemizing = self._n_steps - 2
        with self._replay_memory_client.writer(max_sequence_length=self._n_steps) as writer:
            obs = self._train_env.reset()

            action, reward, done = tf.constant(-1), tf.constant(0.), tf.constant(0.)
            obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=self.OBS_DTYPES[self._env_name]), obs)
            writer.append((action, obs, reward, done))
            for step in it.count(0):
                action = self._epsilon_greedy_policy(obs, epsilon)

                obs, reward, done, info = self._train_env.step(action)
                action = tf.convert_to_tensor(action, dtype=tf.int32)
                reward = tf.convert_to_tensor(reward, dtype=tf.float32)
                done = tf.convert_to_tensor(done, dtype=tf.float32)
                obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x,
                                                                           dtype=self.OBS_DTYPES[self._env_name]),
                                            obs)
                writer.append((action, obs, reward, done))
                if step >= start_itemizing:
                    writer.create_item(table=self._table_name, num_timesteps=self._n_steps, priority=1.)
                if done:
                    break

    def _collect_trajectories_from_episode_atari(self, epsilon):
        """
        Collects trajectories (items) to a buffer.
        A buffer contains items, each item consists of n_steps 'time steps';
        for a regular TD(0) update an item should have 2 time steps.
        One 'time step' contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to the obs, reward prior the obs, if is it done at the current obs.
        """
        lives = 5
        initial_step = True
        repeat_counter = 0
        prev_action = -1
        done = False

        obs = self._train_env.reset()
        # save sequences of n_steps length
        start_itemizing = self._n_steps - 2
        while lives:
            with self._replay_memory_client.writer(max_sequence_length=self._n_steps) as writer:
                if done:
                    # if life was lost, save again the previous step with 'done'
                    pass
                else:
                    # the very first observation (just before reset)
                    action, reward, done = tf.constant(prev_action), tf.constant(0.), tf.constant(0.)
                    obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x,
                                                                               dtype=self.OBS_DTYPES[self._env_name]),
                                                obs)
                writer.append((action, obs, reward, done))
                for step in it.count(0):
                    if initial_step or done:
                        action = 1  # fire, it can probably speed up training
                        initial_step = False
                    else:
                        action = self._epsilon_greedy_policy(obs, epsilon)

                    # to prevent huge sequences of similar actions
                    if action == prev_action:
                        repeat_counter += 1
                    else:
                        repeat_counter = 0
                    prev_action = action

                    obs, reward, done, info = self._train_env.step(action)
                    if info['ale.lives'] < lives:
                        # if life is lost, save done True to the replay buffer
                        done = True
                        lives = info['ale.lives']

                    action = tf.convert_to_tensor(action, dtype=tf.int32)
                    reward = tf.convert_to_tensor(reward, dtype=tf.float32)
                    done = tf.convert_to_tensor(done, dtype=tf.float32)
                    obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x,
                                                                               dtype=self.OBS_DTYPES[self._env_name]),
                                                obs)
                    writer.append((action, obs, reward, done))
                    if step >= start_itemizing:
                        writer.create_item(table=self._table_name, num_timesteps=self._n_steps, priority=1.)
                    if done:
                        break
                    if repeat_counter > self._repeat_limit:
                        done = True  # to fire on the next step
                        break

    def _collect_several_episodes(self, epsilon, n_episodes):
        for i in range(n_episodes):
            self._collect_trajectories_from_episode(epsilon)

    def _collect_until_items_created(self, epsilon, n_items):
        # collect more exp if we do not have enough for a batch
        items_created = self._replay_memory_client.server_info()[self._table_name][5].insert_stats.completed
        while items_created < n_items:
            self._collect_trajectories_from_episode(epsilon)
            items_created = self._replay_memory_client.server_info()[self._table_name][5].insert_stats.completed

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

    def train_collect(self, iterations_number=10000, epsilon=0.1):

        eval_interval = 2000
        target_model_update_interval = 3000
        # self._epsilon = epsilon
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=epsilon,  # initial ε
            decay_steps=iterations_number,
            end_learning_rate=0.1)  # final ε

        weights = None
        mask = None
        rewards = 0
        eval_counter = 0

        for step_counter in range(1, iterations_number + 1):
            # collecting
            items_created = self._replay_memory_client.server_info()[self._table_name][5].insert_stats.completed
            # do not collect new experience if we have not used previous
            # train * X times more than collecting new experience
            if items_created * 20 < self._items_sampled:
                self._epsilon = epsilon_fn(step_counter)
                self._collect_trajectories_from_episode(self._epsilon)

            # dm-reverb returns tensors
            sample = next(self._iterator)
            action, obs, reward, done = sample.data
            key, probability, table_size, priority = sample.info
            experiences, info = (action, obs, reward, done), (key, probability, table_size, priority)
            self._items_sampled += self._sample_batch_size

            self._training_step(*experiences, info=info)
            # print(f"\rTraining. Iteration:{step_counter}; "
            #       f"Items sampled:{self._items_sampled}; Items created:{items_created}", end="")

            if step_counter % eval_interval == 0:
                eval_counter += 1
                # print("\r")
                mean_episode_reward = self._evaluate_episodes_greedy()
                print(f"Iteration:{step_counter:.2f}; "
                      f"Items sampled:{self._items_sampled:.2f}; "
                      f"Items created:{items_created:.2f}; "
                      f"Reward: {mean_episode_reward:.2f}; "
                      f"Epsilon: {self._epsilon:.2f}")
                rewards += mean_episode_reward
                # if mean_episode_reward == 500:
                #     weights = self._model.get_weights()
                #     mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))
                #     checkpoint = None
                #     output_reward = mean_episode_reward
                #     break

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # store weights at the last step
            if step_counter % iterations_number == 0:
                mean_episode_reward = self._evaluate_episodes_greedy(num_episodes=10)
                print(f"Final reward with a model policy is {mean_episode_reward}")
                # print(f"Final epsilon is {self._epsilon}")
                # do not update data in case of sparse net
                # currently the only way to make a sparse net is from a dense net weights and mask
                output_reward = rewards / eval_counter
                if self._is_sparse:
                    weights = self._data['weights']
                    mask = self._data['mask']
                    output_reward = self._data['reward']
                else:
                    weights = self._model.get_weights()
                    mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))

                if self._make_checkpoint:
                    try:
                        checkpoint = self._replay_memory_client.checkpoint()
                    except RuntimeError as err:
                        print(err)
                        checkpoint = err
                else:
                    checkpoint = None

        return weights, mask, output_reward, checkpoint

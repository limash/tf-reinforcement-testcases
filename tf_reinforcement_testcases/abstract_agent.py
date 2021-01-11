import abc
# import time
# import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym

from tf_reinforcement_testcases import models, misc


class Agent(abc.ABC):
    NETWORKS = {'CartPole-v1': models.get_q_mlp,
                'CartPole-v1_duel': models.get_dueling_q_mlp,
                'gym_halite:halite-v0': models.get_halite_q_mlp,
                'gym_halite:halite-v0_duel': models.get_halite_dueling_q_mlp}

    def __init__(self, env_name):
        self._train_env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        self._n_outputs = self._train_env.action_space.n
        self._input_shape = self._train_env.observation_space.shape
        if not self._input_shape:
            space = self._train_env.observation_space
            feature_maps_shape = space['feature_maps'].shape
            scalar_features_shape = space['scalar_features'].shape
            self._input_shape = (feature_maps_shape, scalar_features_shape)
        self._model = None
        self._target_model = None

        self._discount_rate = tf.constant(0.95, dtype=tf.float32)
        self._optimizer = keras.optimizers.Adam(lr=1e-3)
        self._loss_fn = keras.losses.mean_squared_error
        self._replay_memory = None
        self._sample_batch_size = 64

        # parameters for prioritized exp replay
        self._beta = None
        self._beta_increment = None

    def _epsilon_greedy_policy(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self._n_outputs)
        else:
            obs = tf.nest.map_structure(lambda x: x[np.newaxis, :], obs)
            obs = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), obs)
            Q_values = self._model(obs)
            return np.argmax(Q_values[0])

    def _collect_one_step(self, obs, epsilon):
        action = self._epsilon_greedy_policy(obs, epsilon)
        next_obs, reward, done, info = self._train_env.step(action)
        self._replay_memory.append((obs, action, reward, next_obs, done))
        return next_obs, reward, done, info

    def _collect_steps(self, steps, epsilon):
        obs = self._train_env.reset()
        for _ in range(steps):
            obs, reward, done, info = self._collect_one_step(obs, epsilon)
            if done:
                obs = self._train_env.reset()

    def _sample_experiences(self):
        indices = np.random.randint(len(self._replay_memory), size=self._sample_batch_size)
        batch = [self._replay_memory[index] for index in indices]
        observations, actions, rewards, next_observations, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return (observations, actions, rewards, next_observations, dones.astype(int)), None

    def _evaluate_episode(self, epsilon=0):
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

    def _evaluate_episodes(self, num_episodes=3):
        episode_rewards = 0
        for _ in range(num_episodes):
            episode_rewards += self._evaluate_episode()
        return episode_rewards / num_episodes

    @abc.abstractmethod
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        raise NotImplementedError

    def train(self, iterations_number=10000):
        # best_score = 0

        epsilon = 0.1
        step_counter = 0
        eval_interval = 200
        target_model_update_interval = 1000

        weights = None
        mask = None
        mean_episode_reward = 0

        # weights = self._model.get_weights()
        # old_weights = copy.deepcopy(weights)

        obs = self._train_env.reset()
        for iteration in range(iterations_number):
            # epsilon = max(1 - iteration / iterations_number, 0.33)
            # sample and train each step
            # collecting
            # t0 = time.time()
            obs, reward, done, info = self._collect_one_step(obs, epsilon)
            # t1 = time.time()

            experiences, info = self._sample_experiences()
            # dm-reverb returns tensors (in `dataset.take()` case)
            # otherwise convert evrth to tensors before a training step
            if not tf.is_tensor(experiences[-1]):
                experiences = misc.process_experiences(experiences)

            # training
            # t2 = time.time()
            tf_consts_and_vars = (self._discount_rate, self._n_outputs, self._beta, self._beta_increment)
            self._training_step(tf_consts_and_vars, info, *experiences)
            step_counter += 1
            # t3 = time.time()
            if done:
                obs = self._train_env.reset()

            if step_counter % eval_interval == 0:
                mean_episode_reward = self._evaluate_episodes()

                # if mean_episode_reward > best_score:
                #     best_weights = self._model.get_weights()
                #     best_score = mean_episode_reward
                print("\rTraining step: {}, reward: {}, eps: {:.3f}".format(step_counter,
                                                                            mean_episode_reward,
                                                                            epsilon))
                # print(f"Time spend for sampling is {t1 - t0}")
                # print(f"Time spend for training is {t3 - t2}")

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # make a sparse model at the last step
            if step_counter % iterations_number == 0:
                weights = self._model.get_weights()
                mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))
                self._model = models.get_sparse(weights, mask)

                # evaluate a sparse model
                # mean_episode_reward = self._evaluate_episodes()
                # print(f"Episode reward of a sparse net is {mean_episode_reward}")
                # for debugging a sparse model with a batch input
                # self._training_step(tf_consts_and_vars, info, *experiences)

                # old_weights = copy.deepcopy(weights)
                # indx = list(map(lambda x: np.argwhere(np.abs(x) > 0.1), weights))
                # differences = list(map(lambda x, y: x - y, weights, old_weights))
                # diff_indx = list(map(lambda x: np.argwhere(np.abs(x) < 0.1), differences))

        # self._model.set_weights(best_weights)
        return weights, mask, mean_episode_reward

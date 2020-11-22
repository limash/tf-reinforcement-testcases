import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym

from tf_reinforcement_testcases import models


def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


class DQNAgent:
    NETWORKS = {'CartPole-v0': models.get_mlp,
                'CartPole-v1': models.get_mlp,
                'gym_halite:halite-v0': models.get_mlp}

    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._n_outputs = self._env.action_space.n
        input_shape = self._env.observation_space.shape
        self._model = DQNAgent.NETWORKS[env_name](input_shape, self._n_outputs)

        self._best_score = 0
        self._batch_size = 64
        self._discount_rate = 0.95
        self._optimizer = keras.optimizers.Adam(lr=1e-3)
        self._loss_fn = keras.losses.mean_squared_error
        self._replay_memory = deque(maxlen=40000)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)

    def _sample_experiences(self, batch_size):
        indices = np.random.randint(len(self._replay_memory), size=batch_size)
        batch = [self._replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def _collect_one_step(self, state, epsilon):
        action = epsilon_greedy_policy(self._model, state, epsilon)
        next_state, reward, done, info = self._env.step(action)
        self._replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def _collect_steps(self, steps, epsilon):
        obs = self._env.reset()
        step = 0
        while step < steps:
            obs, reward, done, info = self._collect_one_step(obs, epsilon)
            step += 1
            if done:
                self._env.reset()

    def _evaluate_episode(self):
        obs = self._env.reset()
        rewards = []
        while True:
            # by default epsilon=0, so greedy is disabled
            action = epsilon_greedy_policy(self._model, obs)
            obs, reward, done, info = self._env.step(action)
            rewards.append(reward)
            if done:
                break
        return sum(rewards)

    def _training_step(self):
        experiences = self._sample_experiences(self._batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self._model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                           (1 - dones) * self._discount_rate * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self._n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def train(self):
        training_step = 0
        for episode in range(600):
            epsilon = max(1 - episode / 500, 0.01)

            # sample and train each step
            obs = self._env.reset()
            while True:
                # collecting
                t0 = time.time()
                obs, reward, done, info = self._collect_one_step(obs, epsilon)
                t1 = time.time()
                # training
                t2 = time.time()
                self._training_step()
                training_step += 1
                t3 = time.time()
                if done:
                    break

            episode_rewards = []
            for episode_number in range(3):
                episode_rewards.append(self._evaluate_episode())
            mean_episode_reward = sum(episode_rewards)/episode_number+1

            if mean_episode_reward > self._best_score:
                best_weights = self._model.get_weights()
                self._best_score = mean_episode_reward
            # print("\rEpisode: {}, reward: {}, eps: {:.3f}".format(episode, mean_episode_reward, epsilon), end="")
            print("\rTraining step: {}, Episode: {}, reward: {}, eps: {:.3f}".format(training_step,
                                                                                     episode,
                                                                                     mean_episode_reward,
                                                                                     epsilon))
            print(f"Time spend for sampling is {t1-t0}")
            print(f"Time spend for training is {t3-t2}")

        return self._model.set_weights(best_weights)

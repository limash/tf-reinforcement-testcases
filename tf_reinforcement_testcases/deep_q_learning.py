import time
import abc
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym

from tf_reinforcement_testcases import models, misc, storage


class DQNAgent(abc.ABC):
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

    @abc.abstractmethod
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        raise NotImplementedError

    def train(self, iterations_number=10000):
        # best_score = 0

        step_counter = 0
        eval_interval = 200

        obs = self._train_env.reset()
        for iteration in range(iterations_number):
            # epsilon = max(1 - iteration / iterations_number, 0.33)
            epsilon = 0.1
            # sample and train each step
            # collecting
            t0 = time.time()
            obs, reward, done, info = self._collect_one_step(obs, epsilon)
            t1 = time.time()

            experiences, info = self._sample_experiences()
            # dm-reverb returns tensors (in `dataset.take()` case)
            # otherwise convert evrth to tensors before training step
            if not tf.is_tensor(experiences[-1]):
                experiences = misc.process_experiences(experiences)

            # training
            t2 = time.time()
            tf_consts_and_vars = (self._discount_rate, self._n_outputs, self._beta, self._beta_increment)
            self._training_step(tf_consts_and_vars, info, *experiences)
            step_counter += 1
            t3 = time.time()
            if done:
                obs = self._train_env.reset()

            if step_counter % eval_interval == 0:
                if self._target_model and step_counter % 1000 == 0:
                    self._target_model.set_weights(self._model.get_weights())
                episode_rewards = 0
                for episode_number in range(3):
                    episode_rewards += self._evaluate_episode()
                mean_episode_reward = episode_rewards / (episode_number + 1)

                # if mean_episode_reward > best_score:
                #     best_weights = self._model.get_weights()
                #     best_score = mean_episode_reward
                print("\rTraining step: {}, reward: {}, eps: {:.3f}".format(step_counter,
                                                                            mean_episode_reward,
                                                                            epsilon))
                print(f"Time spend for sampling is {t1 - t0}")
                print(f"Time spend for training is {t3 - t2}")

        # self._model.set_weights(best_weights)
        return self._model


class RegularDQNAgent(DQNAgent):

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = DQNAgent.NETWORKS[env_name](self._input_shape, self._n_outputs)
        self._replay_memory = deque(maxlen=40000)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)
        print(f"Random policy reward is {self._evaluate_episode(epsilon=1)}")
        print(f"Untrained policy reward is {self._evaluate_episode()}")

    @tf.function
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        discount_rate, n_outputs, _, _ = tf_consts_and_vars
        next_Q_values = self._model(next_observations)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = (rewards + (tf.constant(1.0) - dones) * discount_rate * max_next_Q_values)
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class FixedQValuesDQNAgent(DQNAgent):
    """
    The agent uses a target model to establish target Q values
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = DQNAgent.NETWORKS[env_name](self._input_shape, self._n_outputs)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
        self._replay_memory = deque(maxlen=40000)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)
        print(f"Random policy reward is {self._evaluate_episode(epsilon=1)}")
        print(f"Untrained policy reward is {self._evaluate_episode()}")

    @tf.function
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        discount_rate, n_outputs, _, _ = tf_consts_and_vars
        next_Q_values = self._target_model(next_observations)  # below everything is similar to the regular DQN
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = (rewards + (tf.constant(1.0) - dones) * discount_rate * max_next_Q_values)
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class DoubleDQNAgent(DQNAgent):
    """
    To establish target Q values the agent uses:
    a model to predict best next actions
    a target model to predict next best Q values corresponding to the best next actions
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = DQNAgent.NETWORKS[env_name](self._input_shape, self._n_outputs)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
        self._replay_memory = deque(maxlen=40000)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)
        print(f"Random policy reward is {self._evaluate_episode(epsilon=1)}")
        print(f"Untrained policy reward is {self._evaluate_episode()}")

    @tf.function
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        discount_rate, n_outputs, _, _ = tf_consts_and_vars
        next_Q_values = self._model(next_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(next_observations) * next_mask), axis=1)
        target_Q_values = (rewards + (tf.constant(1.0) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class DoubleDuelingDQNAgent(DQNAgent):
    """
    Similar to the Double agent, but uses a 'dueling network'
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = DQNAgent.NETWORKS[env_name+'_duel'](self._input_shape, self._n_outputs)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
        self._replay_memory = deque(maxlen=40000)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)
        print(f"Random policy reward is {self._evaluate_episode(epsilon=1)}")
        print(f"Untrained policy reward is {self._evaluate_episode()}")

    @tf.function
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        discount_rate, n_outputs, _, _ = tf_consts_and_vars
        next_Q_values = self._model(next_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(next_observations) * next_mask), axis=1)
        target_Q_values = (rewards + (tf.constant(1.0) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class PriorityDoubleDuelingDQNAgent(DQNAgent):

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = DQNAgent.NETWORKS[env_name+'_duel'](self._input_shape, self._n_outputs)
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
        self._replay_memory = storage.PriorityBuffer(self._sample_batch_size, self._input_shape)

        self._tf_sample_batch_size = tf.constant(self._sample_batch_size, dtype=tf.float32)
        self._beta = tf.Variable(0.4, dtype=tf.float32)
        self._beta_increment = tf.constant(0.0001, dtype=tf.float32)

        # collect some data with a random policy before training
        self._collect_steps(steps=4000, epsilon=1)
        print(f"Random policy reward is {self._evaluate_episode(epsilon=1)}")
        print(f"Untrained policy reward is {self._evaluate_episode()}")

    def _sample_experiences(self):
        return self._replay_memory.sample_batch()

    @tf.function
    def _training_step(self, tf_consts_and_vars, info,
                       observations, actions, rewards, next_observations, dones,
                       ):
        discount_rate, n_outputs, beta, beta_increment = tf_consts_and_vars
        keys = info[0]
        # dm-reverb info has a float64 format, which is incompatible
        info = tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.float32), info[1:])
        probs, table_sizes, priors = info
        beta = tf.reduce_min(tf.stack([tf.constant(1.0), beta + beta_increment]))
        importance_sampling = tf.pow(self._tf_sample_batch_size*probs, -beta)
        max_importance = tf.reduce_max(importance_sampling)
        importance_sampling = importance_sampling / max_importance
        # DDQN part
        next_Q_values = self._model(next_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(next_observations) * next_mask), axis=1)
        target_Q_values = (rewards + (tf.constant(1.0) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            # add importance sampling to the loss function
            loss = tf.reduce_mean(importance_sampling * self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        # calculate new priorities
        absolute_errors = tf.abs(target_Q_values - Q_values)
        absolute_errors = absolute_errors + tf.constant([0.01])  # to avoid skipping some exp
        clipped_errors = tf.minimum(absolute_errors, tf.constant([1.]))  # errors from 0.01 to 1.
        new_priorities = tf.pow(clipped_errors, tf.constant([0.6]))  # increase prob of the less prob priorities
        new_priorities = tf.cast(new_priorities, dtype=tf.float64)
        new_priorities = tf.squeeze(new_priorities, axis=-1)
        self._replay_memory.update_priorities(keys, new_priorities)

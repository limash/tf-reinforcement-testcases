from collections import deque

import ray
import tensorflow as tf
from tensorflow import keras

from tf_reinforcement_testcases import storage
from tf_reinforcement_testcases.abstract_agent import Agent


@ray.remote
class RegularDQNAgent(Agent):

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        if self._data is None:
            self._model = Agent.NETWORKS[env_name](self._input_shape, self._n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=10)
        else:
            self._model = Agent.NETWORKS[env_name](self._data['weights'], self._data['mask'])
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=10)
            # collect date with epsilon greedy policy
            # self._collect_several_episodes(epsilon=self._epsilon, n_episodes=10)

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        next_Q_values = self._model(last_observations)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * max_next_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class FixedQValuesDQNAgent(Agent):
    """
    The agent uses a target model to establish target Q values
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = Agent.NETWORKS[env_name](self._input_shape, self._n_outputs)
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


class DoubleDQNAgent(Agent):
    """
    To establish target Q values the agent uses:
    a model to predict best next actions
    a target model to predict next best Q values corresponding to the best next actions
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = Agent.NETWORKS[env_name](self._input_shape, self._n_outputs)
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


class DoubleDuelingDQNAgent(Agent):
    """
    Similar to the Double agent, but uses a 'dueling network'
    """

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = Agent.NETWORKS[env_name + '_duel'](self._input_shape, self._n_outputs)
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


class PriorityDoubleDuelingDQNAgent(Agent):

    def __init__(self, env_name):
        super().__init__(env_name)

        self._model = Agent.NETWORKS[env_name + '_duel'](self._input_shape, self._n_outputs)
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
        importance_sampling = tf.pow(self._tf_sample_batch_size * probs, -beta)
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

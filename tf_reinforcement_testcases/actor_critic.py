import numpy as np
import tensorflow as tf

from tf_reinforcement_testcases.abstract_agent import Agent
from tf_reinforcement_testcases import models


class ACAgent(Agent):

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        assert not (self._data and self._is_sparse), "A sparse model is not available for actor-critic"

        # train a model from scratch
        if self._data is None:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=10)
        # continue a model training
        elif self._data and not self._is_sparse:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            self._collect_several_episodes(epsilon=self._epsilon, n_episodes=10)

    def _epsilon_greedy_policy(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self._n_outputs)
        else:
            obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs)
            logits, Q_values = self._predict(obs)
            probabilities = tf.nn.softmax(logits)
            return np.argmax(probabilities[0])  # switch to sample categorical

    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        next_logits, next_Q_values = self._model(last_observations)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * max_next_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_logits, all_Q_values = self._model(first_observations)
            probs = tf.nn.softmax(all_logits)
            masked_probs = tf.reduce_sum(probs * mask, axis=1, keepdims=True)
            logs = tf.math.log(masked_probs)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            td_error = tf.stop_gradient(target_Q_values - Q_values)  # to prevent updating critic part by actor
            actor_loss = -1*logs*td_error
            actor_loss = tf.reduce_mean(actor_loss)
            critic_loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

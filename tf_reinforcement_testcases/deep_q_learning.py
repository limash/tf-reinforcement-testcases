import numpy as np
import tensorflow as tf

from tf_reinforcement_testcases import misc, models
from tf_reinforcement_testcases.abstract_agent import Agent

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class RegularDQNAgent(Agent):

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        # train a model from scratch
        if self._data is None:
            self._model = self.NETWORKS[env_name](self._input_shape, self._n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            # self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)
            self._collect_until_items_created(epsilon=self._epsilon, n_items=20000)
        # continue a model training
        elif self._data and not self._is_sparse:
            self._model = self.NETWORKS[env_name](self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            # self._collect_several_episodes(epsilon=self._epsilon, n_episodes=self._sample_batch_size)
            self._collect_until_items_created(epsilon=self._epsilon, n_items=20000)
        # make and train a sparse model from a dense model
        elif self._data and self._is_sparse:
            weights = self._data['weights']
            random_weights = [np.random.uniform(low=-0.03, high=0.03, size=item.shape) for item in weights]
            self._model = models.get_sparse(random_weights, self._data['mask'])
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)

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


class FixedQValuesDQNAgent(RegularDQNAgent):
    """
    The agent uses a target model to establish target Q values
    """

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        if self._is_sparse:
            # make a target model with the weights stored in data
            self._target_model = models.get_sparse(self._data['weights'], self._data['mask'])
            # replace weights of the target model with a weights from the model
            self._target_model.set_weights(self._model.get_weights())
        else:
            self._target_model = self.NETWORKS[env_name](self._input_shape, self._n_outputs)
            self._target_model.set_weights(self._model.get_weights())

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        next_Q_values = self._target_model(last_observations)  # the only difference comparing to the vanilla dqn
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


class DoubleDQNAgent(FixedQValuesDQNAgent):
    """
    To establish target Q values the agent uses:
    a model to predict best next actions
    a target model to predict next best Q values corresponding to the best next actions
    """

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        next_Q_values = self._model(last_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self._n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(last_observations) * next_mask), axis=1)
        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * next_best_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class DoubleDuelingDQNAgent(DoubleDQNAgent):
    """
    Similar to the Double agent, but uses a 'dueling network'
    """

    def __init__(self, env_name, *args, **kwargs):
        # we need a training_step from a DoubleDQNAgent, but initialization from an abstract Agent
        # since dueling changes are in the model, so we need initialize this model
        super(RegularDQNAgent, self).__init__(env_name, *args, **kwargs)

        assert not (self._data and self._is_sparse), "A sparse model is not available for dueling nets"

        # train a model from scratch
        if self._data is None:
            self._model = models.get_dueling_q_mlp(self._input_shape, self._n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)
        # continue a model training
        elif self._data and not self._is_sparse:
            self._model = models.get_dueling_q_mlp(self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            self._collect_several_episodes(epsilon=self._epsilon, n_episodes=self._sample_batch_size)

        self._target_model = models.get_dueling_q_mlp(self._input_shape, self._n_outputs)
        self._target_model.set_weights(self._model.get_weights())


class CategoricalDQNAgent(Agent):

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        min_q_value = 0
        max_q_value = 51
        self._n_atoms = 51
        self._support = tf.linspace(min_q_value, max_q_value, self._n_atoms)
        self._support = tf.cast(self._support, tf.float32)
        cat_n_outputs = self._n_outputs * self._n_atoms
        # train a model from scratch
        if self._data is None:
            self._model = self.NETWORKS(self._input_shape, cat_n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            # self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)
            self._collect_until_items_created(epsilon=1, n_items=self._sample_batch_size)
        # continue a model training
        elif self._data and not self._is_sparse:
            self._model = self.NETWORKS(self._input_shape, cat_n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            self._collect_several_episodes(epsilon=self._epsilon, n_episodes=self._sample_batch_size)
        # make and train a sparse model from a dense model
        elif self._data and self._is_sparse:
            weights = self._data['weights']
            random_weights = [np.random.uniform(low=-0.03, high=0.03, size=item.shape) for item in weights]
            self._model = models.get_sparse(random_weights, self._data['mask'])
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)

        reward = self._evaluate_episodes_greedy(num_episodes=100)
        print(f"Initial reward with a model policy is {reward}")

    def _epsilon_greedy_policy(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self._n_outputs)
        else:
            obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs)
            logits = self._predict(obs)
            logits = tf.reshape(logits, [-1, self._n_outputs, self._n_atoms])
            probabilities = tf.nn.softmax(logits)
            Q_values = tf.reduce_sum(self._support * probabilities, axis=-1)  # Q values expected return
            return np.argmax(Q_values[0])

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        # Part 1: calculate new target (best) Q value distributions (next_best_probs)
        next_logits = self._model(last_observations)
        # reshape to (batch, n_actions, distribution support number of elements (atoms)
        next_logits = tf.reshape(next_logits, [-1, self._n_outputs, self._n_atoms])
        next_probabilities = tf.nn.softmax(next_logits)
        next_Q_values = tf.reduce_sum(self._support * next_probabilities, axis=-1)  # Q values expected return
        # get indices of max next Q values and get corresponding distributions
        max_args = tf.cast(tf.argmax(next_Q_values, 1), tf.int32)[:, None]
        batch_indices = tf.range(tf.cast(self._sample_batch_size, tf.int32))[:, None]
        next_qt_argmax = tf.concat([batch_indices, max_args], axis=-1)  # indices of the target Q value distributions
        next_best_probs = tf.gather_nd(next_probabilities, next_qt_argmax)

        # Part 2: calculate a new but non-aligned support of the target Q value distributions
        batch_support = tf.repeat(self._support[None, :], [self._sample_batch_size], axis=0)
        last_dones = tf.expand_dims(last_dones, -1)
        total_rewards = tf.expand_dims(total_rewards, -1)
        non_aligned_support = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * batch_support

        # Part 3: project the target Q value distributions to the basic (target_support) support
        target_distribution = misc.project_distribution(supports=non_aligned_support,
                                                        weights=next_best_probs,
                                                        target_support=self._support)

        # Part 4: Loss and update
        indices = tf.cast(batch_indices[:, 0], second_actions.dtype)
        reshaped_actions = tf.stack([indices, second_actions], axis=-1)
        with tf.GradientTape() as tape:
            logits = self._model(first_observations)
            logits = tf.reshape(logits, [-1, self._n_outputs, self._n_atoms])
            chosen_action_logits = tf.gather_nd(logits, reshaped_actions)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_distribution,
                                                           logits=chosen_action_logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        return target_distribution, chosen_action_logits


class PriorityCategoricalDQNAgent(CategoricalDQNAgent):
    """
    It is unclear how reverb samples and how to change priorities if an item consist of
    several time steps
    Below implementation is for items of one time step
    """

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        raise NotImplementedError

        # for priory buffer version
        self._importance_sampling = tf.constant(1.)  # should be multiplied by loss

        # priority buffer hyperparameters
        self._n_elements_in_buffer = None
        self._beta = tf.Variable(0.4, dtype=tf.float32)
        self._beta_increment = tf.constant(0.0001, dtype=tf.float32)

    def _training_step(self, actions, observations, rewards, dones, info):
        # update importance sampling hyperparameter
        keys = info[0]
        # dm-reverb info has a float64 format, which is incompatible
        info = tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.float32), info[1:])
        probs, table_sizes, priors = info
        beta = tf.reduce_min(tf.stack([tf.constant(1.0), self._beta + self._beta_increment]))
        importance_sampling = tf.pow(self._n_elements_in_buffer * probs, -beta)
        max_importance = tf.reduce_max(importance_sampling)
        self._importance_sampling = importance_sampling / max_importance

        # the main part
        target_distribution, chosen_action_logits = super()._training_step(actions, observations, rewards, dones, info)
        probabilities = tf.nn.softmax(chosen_action_logits)
        Q_values = tf.reduce_sum(self._support * probabilities, axis=-1)
        next_Q_values = tf.reduce_sum(self._support * target_distribution, axis=-1)

        # calculate new priorities
        absolute_errors = tf.abs(next_Q_values - Q_values)  # td error
        absolute_errors = absolute_errors + tf.constant([0.01])  # to avoid skipping some exp
        clipped_errors = tf.minimum(absolute_errors, tf.constant([1.]))  # errors from 0.01 to 1.
        new_priorities = tf.pow(clipped_errors, tf.constant([0.6]))  # increase prob of the less prob priorities
        new_priorities = tf.cast(new_priorities, dtype=tf.float64)
        new_priorities = tf.squeeze(new_priorities, axis=-1)
        self._replay_memory_client.update_priorities(self._table_name, keys, new_priorities)

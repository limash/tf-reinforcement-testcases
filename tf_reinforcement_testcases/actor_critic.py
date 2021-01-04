import logging

import numpy as np

import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

from tf_reinforcement_testcases import misc


class ACAgent:
    def __init__(self, model):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = 0.99
        self.value_c = 0.5
        self.entropy_c = 1e-4

        self.model = model
        self.model.compile(
            optimizer=optimizers.RMSprop(lr=7e-3),
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * losses.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = losses.SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = losses.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        if self.is_halite:
            obs = misc.process_halite_obs(obs)
        # obs = tf.nest.map_structure(lambda x: x[np.newaxis, :], obs)

        count = 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(int(action))
            if self.is_halite:
                obs = misc.process_halite_obs(obs)

            ep_reward += reward
            count += 1
            if render:
                env.render()
        print(count)
        return ep_reward

    def train(self, env, batch_sz=64, updates=250):
        """
        It samples actions, values, rewards, dones 'batch_sz' number of times

        Args:
            env: gym environment object
            batch_sz: a number of steps to save in a batch
            updates: a number of updates on the batch

        Returns:
            ep_rewards:
        """
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        if self.is_halite:
            next_obs = misc.process_halite_obs(next_obs)

        observations = np.empty((batch_sz,) + next_obs.shape)
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()

                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                if self.is_halite:
                    next_obs = misc.process_halite_obs(next_obs)

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    if self.is_halite:
                        next_obs = misc.process_halite_obs(next_obs)

                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

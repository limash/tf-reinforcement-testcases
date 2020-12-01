import numpy as np
import tensorflow as tf

import reverb


class Buffer:
    def __init__(self, batch_size=64, observations_shape=None):
        self._simple_server = reverb.Server(
            tables=[
                reverb.Table(
                    name='my_table',
                    sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(40000),
                    rate_limiter=reverb.rate_limiters.MinSize(2)),
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None)

        # Initializes the reverb client on the same port as the server.
        self._client = reverb.Client(f'localhost:{self._simple_server.port}')
        self._batch_size = batch_size

        # Sets the sequence length to match the length of the prioritized items
        # inserted into the table.
        self._sequence_length = 1

        maps_shape = tf.TensorShape(observations_shape[0])
        scalars_shape = tf.TensorShape(observations_shape[1])
        observations_shape = (maps_shape, scalars_shape)
        obs_dtypes = tf.nest.map_structure(lambda x: np.float32, observations_shape)

        actions_shape = tf.TensorShape([])
        rewards_shape = tf.TensorShape([])
        dones_shape = tf.TensorShape([])
        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{self._simple_server.port}',
            table='my_table',
            max_in_flight_samples_per_worker=10,
            dtypes=(obs_dtypes, np.int32, np.float32, obs_dtypes, np.int32),
            shapes=(observations_shape, actions_shape, rewards_shape,
                    observations_shape, dones_shape))

        # Batches the data according to the correct sequence length.
        # dataset = dataset.batch(self._sequence_length)
        # Batches sequences together
        self._dataset = dataset.batch(self._batch_size)
        self._iterator = self._dataset.as_numpy_iterator()

    def append(self, trajectory):
        obs, action, reward, next_obs, done = trajectory
        action, reward, done = np.int32(action), np.float32(reward), np.int32(done)
        obs = tf.nest.map_structure(lambda x: np.float32(x), obs)
        next_obs = tf.nest.map_structure(lambda x: np.float32(x), next_obs)
        with self._client.writer(max_sequence_length=self._sequence_length) as writer:
            writer.append((obs, action, reward, next_obs, done))
            writer.create_item(table='my_table', num_timesteps=1, priority=1.)

    def sample_batch(self):
        sample = next(self._iterator)
        obs, action, reward, next_obs, done = sample.data
        return obs, action, reward, next_obs, done

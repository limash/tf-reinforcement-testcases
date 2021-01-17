import numpy as np
import tensorflow as tf

import reverb


def initialize_dataset(server_port, table_name, observations_shape, batch_size, n_steps):
    """
    batch_size in fact equals min size of a buffer
    """
    # if there are many dimensions assume halite
    if len(observations_shape) > 1:
        maps_shape = tf.TensorShape(observations_shape[0])
        scalars_shape = tf.TensorShape(observations_shape[1])
        observations_shape = (maps_shape, scalars_shape)
    else:
        observations_shape = tf.nest.map_structure(lambda x: tf.TensorShape(x), observations_shape)

    actions_shape = tf.TensorShape([])
    rewards_shape = tf.TensorShape([])
    dones_shape = tf.TensorShape([])

    obs_dtypes = tf.nest.map_structure(lambda x: np.float32, observations_shape)

    dataset = reverb.ReplayDataset(
        server_address=f'localhost:{server_port}',
        table=table_name,
        max_in_flight_samples_per_worker=10,
        dtypes=(np.int32, obs_dtypes, np.float32, np.float32),
        shapes=(actions_shape, observations_shape, rewards_shape, dones_shape))

    dataset = dataset.batch(n_steps)
    dataset = dataset.batch(batch_size)

    return dataset


class UniformBuffer:
    # Only server.port and a table name are required to make a client and a dataset
    # thus, theoretically a function returning a port number and a table name
    # should be enough, but it does not work;
    # a server object apparently should be 'alive', above or same lvl of all objects using it
    def __init__(self,
                 min_size: int = 64,
                 max_size: int = 40000):

        self._min_size = min_size
        self._table_name = 'uniform_table'
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_name,
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size)),
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None)
        self._dataset = None

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def server_port(self) -> int:
        return self._server.port


class PriorityBuffer:
    def __init__(self, batch_size=64, observations_shape=None, size=40000):
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name='priority_table',
                    sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(size),
                    rate_limiter=reverb.rate_limiters.MinSize(2)),
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None)

        # Initializes the reverb client on the same port as the server.
        self._client = reverb.Client(f'localhost:{self._server.port}')
        self._tf_client = reverb.TFClient(f'localhost:{self._server.port}')
        self._batch_size = batch_size

        # Sets the sequence length to match the length of the prioritized items
        # inserted into the table.
        self._sequence_length = 1

        # if there are many dimensions assume halite
        if len(observations_shape) > 1:
            maps_shape = tf.TensorShape(observations_shape[0])
            scalars_shape = tf.TensorShape(observations_shape[1])
            observations_shape = (maps_shape, scalars_shape)
        else:
            observations_shape = tf.nest.map_structure(lambda x: tf.TensorShape(x), observations_shape)

        actions_shape = tf.TensorShape([])
        rewards_shape = tf.TensorShape([])
        dones_shape = tf.TensorShape([])

        obs_dtypes = tf.nest.map_structure(lambda x: np.float32, observations_shape)

        dataset = reverb.ReplayDataset(
            server_address=f'localhost:{self._server.port}',
            table='priority_table',
            max_in_flight_samples_per_worker=10,
            dtypes=(obs_dtypes, np.int32, np.float32, obs_dtypes, np.float32),
            shapes=(observations_shape, actions_shape, rewards_shape,
                    observations_shape, dones_shape))

        self._dataset = dataset.batch(self._batch_size)
        # self._iterator = self._dataset.as_numpy_iterator()

    def append(self, trajectory):
        obs, action, reward, next_obs, done = trajectory
        action, reward, done = np.int32(action), np.float32(reward), np.float32(done)
        obs = tf.nest.map_structure(lambda x: np.float32(x), obs)
        next_obs = tf.nest.map_structure(lambda x: np.float32(x), next_obs)
        trajectory = obs, action, reward, next_obs, done
        # put all new trajectories with priority 1
        # it differs from the original article "Prioritized Experience Replay"
        self._client.insert(trajectory, {'priority_table': 1.})
        # with self._client.writer(max_sequence_length=self._sequence_length) as writer:
        #     writer.append((obs, action, reward, next_obs, done))
        #     writer.create_item(table='priority_table', num_timesteps=1, priority=1.)

    def sample_batch(self):
        for sample in self._dataset.take(1):
            obs, action, reward, next_obs, done = sample.data
            key, probability, table_size, priority = sample.info
        # sample = next(self._iterator)
        return (obs, action, reward, next_obs, done), (key, probability, table_size, priority)

    def update_priorities(self, keys, priorities):
        self._tf_client.update_priorities('priority_table', keys, priorities)

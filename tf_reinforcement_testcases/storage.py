import tensorflow as tf

import reverb


def initialize_dataset(server_port, table_name, observations_shape, batch_size, n_steps):
    """
    batch_size in fact equals min size of a buffer
    """
    # if len(observations_shape) > 1:
    #     # maps_shape = tf.TensorShape(observations_shape[0])
    #     # scalars_shape = tf.TensorShape(observations_shape[1])
    #     # observations_shape = (maps_shape, scalars_shape)
    # else:
    #     observations_shape = tf.nest.map_structure(lambda x: tf.TensorShape(x), observations_shape)
    observations_shape = tf.TensorShape(observations_shape)

    actions_shape = tf.TensorShape([])
    rewards_shape = tf.TensorShape([])
    dones_shape = tf.TensorShape([])

    obs_dtypes = tf.nest.map_structure(lambda x: tf.float32, observations_shape)

    dataset = reverb.ReplayDataset(
        server_address=f'localhost:{server_port}',
        table=table_name,
        max_in_flight_samples_per_worker=10,
        dtypes=(tf.int32, obs_dtypes, tf.float32, tf.float32),
        shapes=(actions_shape, observations_shape, rewards_shape, dones_shape))

    dataset = dataset.batch(n_steps)
    dataset = dataset.batch(batch_size)

    return dataset


class UniformBuffer:
    def __init__(self,
                 min_size: int = 64,
                 max_size: int = 100000):

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
    def __init__(self,
                 min_size: int = 64,
                 max_size: int = 100000):
        self._min_size = min_size
        self._table_name = 'priority_table'
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_name,
                    sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size)),
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None)

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def server_port(self) -> int:
        return self._server.port

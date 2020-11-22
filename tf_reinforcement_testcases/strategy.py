import tensorflow as tf


def define_strategy():
    """
    Defines a calculation strategy depending on which processors
    are present.

    :return: A tensorflow strategy class object
    """
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy for hardware
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.master())
    elif len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy(gpus)  # for 1 to multiple GPUs
        print('Running on ', len(gpus), ' GPU(s) ')
    else:
        strategy = tf.distribute.get_strategy()  # works on CPU and single GPU
        print('Running on CPU')

    # How many accelerators do we have ?
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy

    # For distributed computing, the batch size and learning rate need to
    # be adjusted: num replcas is 8 on a single TPU or N when runing on N GPUs.
    # global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    # learning_rate = LEARNING_RATE * strategy.num_replicas_in_sy


if __name__ == "__main__":
    print("It is a module file")
    item = define_strategy()
    print(type(item))

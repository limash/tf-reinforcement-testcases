import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import kaggle_environments.envs.halite.helpers as hh

from gym_halite.envs.halite_env import get_scalar_features, get_feature_maps


def process_experiences(experiences):
    observations, actions, rewards, next_observations, dones = experiences

    try:
        observations = tf.nest.map_structure(
            lambda *x: tf.convert_to_tensor(x, dtype=tf.float32), *observations)
        next_observations = tf.nest.map_structure(
            lambda *x: tf.convert_to_tensor(x, dtype=tf.float32), *next_observations)
    except ValueError:
        observations = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, dtype=tf.float32), observations)
        next_observations = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, dtype=tf.float32), next_observations)

    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    return observations, actions, rewards, next_observations, dones


def get_halite_agent(policy):
    """halite agent """
    def halite_agent(obs, config):
        from collections import OrderedDict

        directions = [hh.ShipAction.NORTH,
                      hh.ShipAction.SOUTH,
                      hh.ShipAction.WEST,
                      hh.ShipAction.EAST]

        board = hh.Board(obs, config)
        me = board.current_player

        skalar_features = get_scalar_features(board)
        skalar_features = skalar_features[np.newaxis, ...]
        skalar_features = tf.convert_to_tensor(skalar_features, dtype=tf.float32)
        feature_maps = get_feature_maps(board)
        feature_maps = feature_maps[np.newaxis, ...]
        feature_maps = tf.convert_to_tensor(feature_maps, dtype=tf.float32)
        obs = OrderedDict({'feature_maps': feature_maps, 'scalar_features': skalar_features})

        Q_values = policy(obs)
        action_number = np.argmax(Q_values.numpy()[0])
        try:
            me.ships[0].next_action = directions[action_number]
        except IndexError:
            pass
        return me.next_actions
    return halite_agent


def get_optimizer(steps_per_epoch, number_of_epochs,
                  optimizer_name):
    """ Returns tf optimizer, possible optimizer_name values:
    'adam', 'sgd'.

    Args:
        steps_per_epoch (int): size of an epoch
        number_of_epochs (int): amount of epochs after the learning rate decreases
        optimizer_name (str): a name of a optimizer

    Returns:
        obj: A tensorflow optimizer object
    """

    # Hyperbolically decrease the learning rate
    # to 1/2 of the base rate at number_of_epochs,
    # to 1/3 at 2*number_of_epochs and so on
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.0001,
        decay_steps=steps_per_epoch * number_of_epochs,
        decay_rate=1,
        staircase=False
    )

    optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'RMSprop': tf.keras.optimizers.RMSprop,
    }

    opt = optimizers[optimizer_name](lr_schedule)
    return opt


def get_lrfn(replicas_in_sync):
    """
    A learning rate function.

    :param replicas_in_sync: a number of TPU cores
    :return: A callable - learning rate function
    """
    LR_START = 0.00001
    LR_MIN = 0.00001
    LR_MAX = 0.00005 * replicas_in_sync
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr

    return lrfn


def get_callbacks(patience, replicas_in_sync,
                  logdir, name):
    """ Define callbacks.

    Args:
        patience (int): amount of epochs of patience
        replicas_in_sync (int): number of cores
        logdir (str): directory to save logs
        name (str): name of the model

    Returns:
       list: A callbacks list
    """
    lrfn = get_lrfn(replicas_in_sync)
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
        tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False),
        tf.keras.callbacks.TensorBoard(logdir + '/' + name)
    ]


def draw_learning_rate(epochs):
    rng = [i for i in range(epochs)]
    y = [get_lrfn(1)(x) for x in rng]
    plt.plot(rng, y)
    plt.show()
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


def plot_2d_array(array, name):
    fig = plt.figure(1)
    # make a color map of fixed colors
    # cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(  # noqa
        'my_colormap',
        ['blue', 'black', 'red'],
        256
    )
    # bounds = [-6, -2, 2, 6]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)  # noqa

    # tell imshow about color map so that only set colors are used
    # img = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    img = plt.imshow(array, interpolation='nearest', cmap=cmap)

    # make a color bar
    # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-5, 0, 5])
    plt.colorbar(img)
    # plt.show()
    fig.savefig("data/"+name+".png")

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import ray

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


# @ray.remote(num_gpus=1)
def use_gpu():
    """
    Call to check ids of available GPUs:
    ray.init()
    use_gpu.remote()
    """
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))


def check_halite_agent(model):
    from kaggle_environments import make

    board_size = 5
    starting_halite = 5000
    env = make('halite',
               configuration={"size": board_size,
                              "startingHalite": starting_halite},
               debug=True)
    trainer = env.train([None])
    obs = trainer.reset()

    halite_agent = get_halite_agent(model)
    return halite_agent(obs, env.configuration)


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

        scalar_features = get_scalar_features(board)
        scalar_features = scalar_features[np.newaxis, ...]
        scalar_features = tf.convert_to_tensor(scalar_features, dtype=tf.float32)
        feature_maps = get_feature_maps(board)
        feature_maps = feature_maps[np.newaxis, ...]
        feature_maps = tf.convert_to_tensor(feature_maps, dtype=tf.float32)
        obs = OrderedDict({'feature_maps': feature_maps, 'scalar_features': scalar_features})

        Q_values = policy(obs)
        action_number = np.argmax(Q_values.numpy()[0])
        try:
            me.ships[0].next_action = directions[action_number]
        except IndexError:
            pass
        return me.next_actions
    return halite_agent


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
    plt.close(fig)

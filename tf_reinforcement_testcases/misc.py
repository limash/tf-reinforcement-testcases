import numpy as np
import tensorflow as tf


def process_halite_obs(inputs):
    flat_map = inputs['feature_maps'].flatten()
    scalar_features = inputs['scalar_features']
    all_inputs = np.concatenate((flat_map, scalar_features))
    return all_inputs


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
    import matplotlib.pyplot as plt

    rng = [i for i in range(epochs)]
    y = [get_lrfn(1)(x) for x in rng]
    plt.plot(rng, y)
    plt.show()
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


if __name__ == "__main__":
    print("It is a module file")
    draw_learning_rate(100)

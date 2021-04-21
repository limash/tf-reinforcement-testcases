import tensorflow as tf


CONF_DQN = {
    "agent": "regular",
    "environment": "CartPole-v1",
    "iterations_number": 20000,
    "eval_interval": 2000,
    "batch_size": 64,
    "buffer_size": 1000000,
    "n_steps": 2,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": .1,  # 1 means random sampling, for sampling before training
    "init_n_samples": 1000,
    "start_epsilon": .1,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-5),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(.999, dtype=tf.float32)
}


CONF_RandomDQN = {
    "agent": "randomDQN",
    "iterations_number": 20000,
    "eval_interval": 2000,
    "batch_size": 64,
    "buffer_size": 1000000,
    "n_steps": 4,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": .5,  # 1 means random sampling, for sampling before training
    "init_n_samples": 1000,
    "start_epsilon": .5,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-6),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(.999, dtype=tf.float32)
}


CONF_CategoricalDQN = {
    "agent": "categoricalDQN",
    "iterations_number": 20000,
    "eval_interval": 2000,
    "batch_size": 64,
    "buffer_size": 1000000,
    "n_steps": 4,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": .5,  # 1 means random sampling, for sampling before training
    "init_n_samples": 1000,
    "start_epsilon": .5,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-6),
    # "optimizer": tf.keras.optimizers.RMSprop(lr=1.e-4, rho=0.95, momentum=0.0,
    #                                          epsilon=0.00001, centered=True),
    "loss": None,  # it is hard-coded in the categorical algorithm
    "discount_rate": tf.constant(.99, dtype=tf.float32)
}

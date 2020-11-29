from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers


class ProbabilityDistribution(tf.keras.Model, ABC):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class ActorCriticModel(tf.keras.Model, ABC):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # joint layers
        self.hidden1 = layers.Dense(1024, activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(0.01))
        self.hidden2 = layers.Dense(1024, activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(0.01))

        # critic predicts an expected value
        self.hidden_value = layers.Dense(1024, activation='relu',
                                         kernel_regularizer=keras.regularizers.l2(0.01))
        self.value = layers.Dense(1, name='value')

        # actor predicts actions
        # Logits are unnormalized log probabilities.
        self.hidden_logits = layers.Dense(1024, activation='relu',
                                          kernel_regularizer = keras.regularizers.l2(0.01))
        self.logits = layers.Dense(num_actions, name='policy_logits')

        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        # todo: move convertion out of here
        x = tf.convert_to_tensor(inputs)

        # joint layers
        y = self.hidden1(x)
        z = self.hidden2(x)

        # fork for an expected value and logits
        hidden_val = self.hidden_value(y)
        val = self.value(hidden_val)

        hidden_logs = self.hidden_logits(z)
        logs = self.logits(hidden_logs)

        return logs, val

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):

        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs, **kwargs):

        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def get_resnet33(inputs):
    """
    Makes a ResNet with 33 layers
    Args:
        inputs: keras.layers.Input() object

    Returns:
        a keras layer - resnet
    """
    x = keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        x = ResidualUnit(filters, strides=strides)(x)
        prev_filters = filters

    # since the last res units have 512 feature maps, x should have length 512 after global pooling
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Flatten()(x)
    return x


def get_halite_net(map_shape):
    """
    A halite net.

    Args:
        map_shape: [x, y, number of layers (feature maps)]
    """
    input_map = keras.layers.Input(shape=map_shape, name="map_input")
    return input_map, get_resnet33(input_map)


def get_halite_critic_keras_model(map_shape, scalar_features_length, actions_length):
    """

    Args:
        map_shape: [x, y, number of features layers]
        scalar_features_length: number of scalar features
        actions_length: it should be 1

    Returns:
        keras model which predicts q values - estimated reward
    """
    input_map, conv_net_output = get_halite_net(map_shape)
    input_scalar = keras.layers.Input(shape=scalar_features_length, name="scalar_features_input")
    input_actions = keras.layers.Input(shape=actions_length, name="actions_input")

    concat = keras.layers.concatenate([conv_net_output, input_scalar, input_actions])
    dense1 = keras.layers.Dense(1024, activation="relu")(concat)
    dense2 = keras.layers.Dense(1024, activation="relu")(dense1)
    output = keras.layers.Dense(1, name="output")(dense2)

    model = keras.Model(inputs=[input_map, input_scalar, input_actions], outputs=[output])
    return model


def get_halite_actor_keras_model(map_shape, scalar_features_length):
    """

    Args:
        map_shape: [x, y, number of features layers]
        scalar_features_length: number of scalar features

    Returns:
        keras model which predicts actions, more precisely,
        it should predict parameters for a tanh-squashed MultivariateNormalDiag distribution
    """
    input_map, conv_net_output = get_halite_net(map_shape)
    input_scalar = keras.layers.Input(shape=scalar_features_length, name="scalar_features_input")

    concat = keras.layers.concatenate([conv_net_output, input_scalar])
    dense1 = keras.layers.Dense(1024, activation="relu")(concat)
    dense2 = keras.layers.Dense(1024, activation="relu")(dense1)
    output = None  # to add projection

    model = keras.Model(inputs=[input_map, input_scalar], outputs=[output])
    return model


def get_q_mlp(input_shape, n_outputs):
    model = keras.models.Sequential([
        keras.layers.Dense(100, activation="relu", input_shape=input_shape),
        # keras.layers.Dense(128, activation="elu"),
        keras.layers.Dense(n_outputs)
    ])
    return model


def get_dueling_q_mlp(input_shape, n_outputs):
    input_states = layers.Input(shape=input_shape)
    x = layers.Dense(100, activation="elu")(input_states)
    # x = layers.Dense(32, activation="elu")(x)
    state_values = layers.Dense(1)(x)
    raw_advantages = layers.Dense(n_outputs)(x)
    advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=1, keepdims=True)
    Q_values = state_values + advantages
    model = keras.Model(inputs=[input_states], outputs=[Q_values])
    return model


def get_halite_q_mlp(input_shape, n_outputs):
    feature_maps_shape, scalar_features_shape = input_shape
    # create inputs
    feature_maps_input = layers.Input(shape=feature_maps_shape, name="feature_maps")
    flatten_feature_maps = layers.Flatten()(feature_maps_input)
    scalar_feature_input = layers.Input(shape=scalar_features_shape, name="scalar_features")
    # concatenate inputs
    x = layers.Concatenate(axis=-1)([flatten_feature_maps, scalar_feature_input])
    # the stem
    stem_kernel_initializer = tf.keras.initializers.variance_scaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'
    )
    output_kernel_initializer = tf.keras.initializers.random_uniform(
        minval=-0.03, maxval=0.03
    )
    output_bias_initializer = tf.keras.initializers.constant(-0.2)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    output = keras.layers.Dense(n_outputs, name="output",
                                kernel_initializer=output_kernel_initializer,
                                bias_initializer=output_bias_initializer)(x)
    # the model
    model = keras.Model(inputs=[feature_maps_input, scalar_feature_input],
                        outputs=[output])
    return model


def get_halite_dueling_q_mlp(input_shape, n_outputs):
    feature_maps_shape, scalar_features_shape = input_shape
    # create inputs
    feature_maps_input = layers.Input(shape=feature_maps_shape, name="feature_maps")
    flatten_feature_maps = layers.Flatten()(feature_maps_input)
    scalar_feature_input = layers.Input(shape=scalar_features_shape, name="scalar_features")
    # concatenate inputs
    x = layers.Concatenate(axis=-1)([flatten_feature_maps, scalar_feature_input])
    # the stem
    stem_kernel_initializer = tf.keras.initializers.variance_scaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'
    )
    output_kernel_initializer = tf.keras.initializers.random_uniform(
        minval=-0.03, maxval=0.03
    )
    output_bias_initializer = tf.keras.initializers.constant(-0.2)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=stem_kernel_initializer)(x)
    state_values = keras.layers.Dense(1,
                                      kernel_initializer=output_kernel_initializer,
                                      bias_initializer=output_bias_initializer)(x)
    raw_advantages = keras.layers.Dense(n_outputs,
                                        kernel_initializer=output_kernel_initializer,
                                        bias_initializer=output_bias_initializer)(x)
    advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=1, keepdims=True)
    Q_values = state_values + advantages
    # the model
    model = keras.Model(inputs=[feature_maps_input, scalar_feature_input],
                        outputs=[Q_values])
    return model

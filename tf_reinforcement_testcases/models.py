# move all imports inside functions to use ray.remote multitasking

def get_q_mlp(input_shape, n_outputs):
    """
    Return Q values of actions
    """
    from tensorflow import keras

    model = keras.models.Sequential([
        keras.layers.Dense(5, activation="relu", input_shape=input_shape),
        # keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(n_outputs)
    ])
    return model


def get_actor_mlp(input_shape, n_outputs):
    """
    Return probabilities of actions (n_outputs)
    """
    from tensorflow import keras

    model = keras.models.Sequential([
        keras.layers.Dense(5, activation="relu", input_shape=input_shape),
        keras.layers.Dense(n_outputs, activation="softmax")
    ])
    return model


def get_dueling_q_mlp(input_shape, n_outputs):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    input_states = layers.Input(shape=input_shape)
    x = layers.Dense(100, activation="relu")(input_states)
    # x = layers.Dense(32, activation="relu")(x)
    state_values = layers.Dense(1)(x)
    raw_advantages = layers.Dense(n_outputs)(x)
    advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=1, keepdims=True)
    Q_values = state_values + advantages
    model = keras.Model(inputs=[input_states], outputs=[Q_values])
    return model


def get_halite_q_mlp(input_shape, n_outputs):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as layers

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
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as layers

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


def get_sparse(weights_in, mask_in):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    class DenseMaskedLayer(keras.layers.Layer):
        def __init__(self, w_init, b_init, mask):
            super(DenseMaskedLayer, self).__init__()

            # w size is (input_dimensions, units)
            self._w = tf.Variable(initial_value=w_init, trainable=True)
            self._b = tf.Variable(initial_value=b_init, trainable=True)
            # mask size is similar to w size
            self._mask = tf.constant(mask, dtype=tf.float32)

        def call(self, inputs, **kwargs):
            return tf.matmul(inputs, self._w * self._mask) + self._b

    class SparseLayer(keras.layers.Layer):
        def __init__(self, w_init, b_init, mask):
            super(SparseLayer, self).__init__()
            # w size is (input_dimensions, units)

            bool_mask = mask.astype(np.bool)
            self._w = []
            self._mask = []
            num_neurons = self._num_neurons = w_init.shape[-1]
            for i in range(num_neurons):
                weights = w_init[:, i]
                masked_weights = weights[bool_mask[:, i]]
                # making a column vector, it is necessary for tf matrix multiplication
                masked_weights_column = masked_weights[..., None]
                self._w.append(tf.Variable(initial_value=masked_weights_column, trainable=True))
                self._mask.append(tf.constant(bool_mask[:, i], dtype=tf.bool))

            self._b = tf.Variable(initial_value=b_init, trainable=True)

        def call(self, inputs, **kwargs):
            neurons = []
            for i in range(self._num_neurons):
                # reshape mask to (batch_size x inputs_size)
                mask = tf.broadcast_to(self._mask[i], [inputs.shape[0], self._mask[i].shape[0]])
                # mask inputs
                masked_inputs = tf.boolean_mask(inputs, mask)
                # restore dimensions after masking
                reshaped_masked_inputs = tf.reshape(
                    masked_inputs, [inputs.shape[0], tf.reduce_sum(tf.cast(self._mask[i], tf.int32)).numpy()])
                # matrix multiplication for one neuron
                neuron = tf.matmul(reshaped_masked_inputs, self._w[i]) + self._b[i]
                neurons.append(neuron)

            result = tf.stack([*neurons], axis=1)
            result = result[..., 0]  # all except the last dimension
            return result

    class SparseMLP(keras.Model):
        def __init__(self, weights, mask):
            super(SparseMLP, self).__init__()

            number_of_layers = int(len(weights) / 2)
            self._main_layers = []
            for i in range(0, number_of_layers):
                self._main_layers.append(SparseLayer(weights[i * 2], weights[i * 2 + 1], mask[i * 2]))
                # do not add activation on the last layer
                if i != number_of_layers - 1:
                    self._main_layers.append(keras.layers.Activation("relu"))

        def call(self, inputs, **kwargs):
            Z = inputs
            for layer in self._main_layers:
                Z = layer(Z)
            return Z

    model = SparseMLP(weights_in, mask_in)
    return model


# class ResidualUnit(keras.layers.Layer):
#     def __init__(self, filters, strides=1, activation="relu", **kwargs):
#
#         super().__init__(**kwargs)
#         self.activation = keras.activations.get(activation)
#         self.main_layers = [
#             keras.layers.Conv2D(filters, 3, strides=strides,
#                                 padding="same", use_bias=False),
#             keras.layers.BatchNormalization(),
#             self.activation,
#             keras.layers.Conv2D(filters, 3, strides=1,
#                                 padding="same", use_bias=False),
#             keras.layers.BatchNormalization()
#         ]
#         self.skip_layers = []
#         if strides > 1:
#             self.skip_layers = [
#                 keras.layers.Conv2D(filters, 1, strides=strides,
#                                     padding="same", use_bias=False),
#                 keras.layers.BatchNormalization()
#             ]
#
#     def call(self, inputs, **kwargs):
#
#         Z = inputs
#         for layer in self.main_layers:
#             Z = layer(Z)
#         skip_Z = inputs
#         for layer in self.skip_layers:
#             skip_Z = layer(skip_Z)
#         return self.activation(Z + skip_Z)
#
#
# def get_resnet33(inputs):
#     """
#     Makes a ResNet with 33 layers
#     Args:
#         inputs: keras.layers.Input() object
#
#     Returns:
#         a keras layer - resnet
#     """
#     x = keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(inputs)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#
#     prev_filters = 64
#     for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
#         strides = 1 if filters == prev_filters else 2
#         x = ResidualUnit(filters, strides=strides)(x)
#         prev_filters = filters
#
#     # since the last res units have 512 feature maps, x should have length 512 after global pooling
#     x = keras.layers.GlobalAvgPool2D()(x)
#     x = keras.layers.Flatten()(x)
#     return x
#
#
# def get_halite_net(map_shape):
#     """
#     A halite net.
#
#     Args:
#         map_shape: [x, y, number of layers (feature maps)]
#     """
#     input_map = keras.layers.Input(shape=map_shape, name="map_input")
#     return input_map, get_resnet33(input_map)

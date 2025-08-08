"""

File Properties:
* Project Name : DL Individual Project
* File Name : pre_encoder.py
* Programmer : Yu Du
* Last Update : April 21, 2025

========================================

Functions:
* PreEncoder::__init__ -- Initialize the pre-coder class
* PreEncoder::add_positional_encoding -- Added sinusoidal position encoding (not trainable)
* PreEncoder::call -- Pre-coder call function (automatically triggered when the model is called)

"""


import math
import tensorflow as tf


class PreEncoder(tf.keras.layers.Layer):

    def __init__(self, num_filters=384, **kwargs):

        """
        Initialize the pre-coder class

        INPUT:
        - [num_filters]: number of convolutional layer output channels
        """

        # Call parent class initialization
        super(PreEncoder, self).__init__(**kwargs)

        # The first one-dimensional convolutional layer
        self.conv1 = tf.keras.layers.Conv1D(
            filters=num_filters,                                    # Set the number of output channels of the convolutional layer
            kernel_size=3,                                          # Set the kernel size
            strides=1,                                              # Set the sliding step size of the kernel
            padding='same',                                         # Set the convolutional layer output mode (same size as input)
            data_format="channels_last",                            # Set the data format: 3D tensor with shape = (batch_shape, steps, channels)
            activation=tf.nn.gelu,                                  # Set the activation function
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)       # Set L2 regularization to avoid overfitting
        )

        # The second one-dimensional convolutional layer
        self.conv2 = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=3,
            strides=2,
            padding='same',
            data_format="channels_last",
            activation=tf.nn.gelu,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

    def add_positional_encoding(self, x):

        """
        Added sinusoidal position encoding (not trainable)

        INPUT:
        - [x]: features after convolution, shape=(batch, time_steps, channels)

        OUTPUT:
        - [x + pos_encoding]: features with position encoding
        """

        # Extract dynamic shape
        _, time_steps, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Construct a time position sequence [0, 1, 2, ..., time_steps-1], shape = (time_steps, 1)
        position = tf.range(time_steps, dtype=tf.float32)[:, tf.newaxis]

        # Used to control the frequency on different channels (equally spaced logarithms)
        div_term = tf.exp(
            tf.range(0, channels, 2, dtype=tf.float32) * (-math.log(10000.0) / tf.cast(channels, tf.float32)))

        # Compute sin and cos components, shape = (time_steps, channels//2)
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)

        # Interleave sin and cos
        pos_encoding = tf.reshape(tf.stack([pe_sin, pe_cos], axis=-1), [time_steps, channels])

        # If the number of channels is odd, truncate to make it consistent with x
        pos_encoding = pos_encoding[:, :channels]

        # Expand the batch dimension to [1, time_steps, channels] to broadcast to x
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        return x + pos_encoding

    def call(self, inputs):

        """
        Pre-coder call function (automatically triggered when the model is called)

        INPUT:
        - [inputs]: input features, shape = (batch_size, time_steps, channels)

        OUTPUT:
        - [x]: features after adding position encoding
        """

        # First convolution
        x = self.conv1(inputs)

        # Second convolution
        x = self.conv2(x)

        # Add sinusoidal position encoding
        x = self.add_positional_encoding(x)

        return x
 

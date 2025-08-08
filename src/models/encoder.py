"""

File Properties:
* Project Name : DL Individual Project
* File Name : encoder.py
* Programmer : Yu Du
* Last Update : April 21, 2025

========================================

Functions:
* Encoder::__init__ -- Initialize the various sub-layers of the encoder module
* Encoder::call -- Assembling the encoder sublayer

"""


import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, d_model=384, num_heads=8, ff_dim=512, dropout_rate=0.4):

        """
        Initialize the various sub-layers of the encoder module

        INPUT:
        - [d_model]: input/output feature dimensions
        - [num_heads]: number of heads for multi-head attention
        - [ff_dim]: hidden layer dimensions of feedforward networks
        - [dropout_rate]: Dropout rate
        """

        # Initialize the parent class
        super().__init__()

        # ----- Sublayer 1: Multi-head self-attention -----

        # Initialize the multi-head self-attention layer: query, key, and value are all input x itself
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,                                        # Number of attention heads
            key_dim=d_model                                             # The feature dimension of each head
        )

        # Initialize Dropout to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        # LayerNorm after residual connection (attention branch)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # ----- Sublayer 2: Feedforward Network -----

        # Feedforward network: Dense → GELU → Dense
        self.ffn = tf.keras.Sequential([

            # First level: Dimensional enhancement + activation
            tf.keras.layers.Dense(ff_dim, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

            # Second layer: restore to d_model
            tf.keras.layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

        ])

        # Initialize Dropout to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # LayerNorm after residual connection (feedforward network branch)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, x, training=False):

        """
        Assembling the encoder sublayer

        INPUT:
        - [x]: input tensor, shape = (batch_size, time_steps, d_model)
        - [training]: whether it is training mode (used to control dropout behavior)

        OUTPUT:
        - [x]: output tensor, shape is the same as input
        """

        # ----- Sublayer 1: Multi-head self-attention -----

        # Calculate self-attention (q=k=v=x), output shape = (batch, time, d_model)
        attn_output = self.attn(x, x, x)

        # Using Dropout to prevent overfitting
        attn_output = self.dropout1(attn_output, training=training)

        # Residual Connection + LayerNorm
        x = self.norm1(x + attn_output)

        # ----- Sublayer 2: Feedforward Network -----

        # Feature transformation through two-layer fully connected network
        ffn_output = self.ffn(x)

        # Using Dropout to prevent overfitting
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual Connection + LayerNorm
        x = self.norm2(x + ffn_output)

        return x
    

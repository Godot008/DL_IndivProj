"""

File Properties:
* Project Name : DL Individual Project
* File Name : multi_layer_audio_classifier.py
* Programmer : Yu Du
* Last Update : May 4, 2025

========================================

Functions:
* AudioClassifier::__init__ -- Initialize the classifier (output layer)
* AudioClassifier::call -- Combining the sub-layers of the classifier

"""


import tensorflow as tf


class AudioClassifier(tf.keras.layers.Layer):

    def __init__(self, num_classes=10, hidden_dim=128, dropout_rate=0.4):

        """
        Initialize the classifier (output layer)

        INPUT:
        - [num_classes]: the number of categories for classification (10 categories for the GTZAN dataset)
        - [hidden_dim]: middle hidden layer size
        - [dropout_rate]: Dropout rate
        """

        super().__init__()

        # The first fully connected layer (with activation) extracts features
        self.fc1 = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        # Dropout, increase regularization, and prevent overfitting
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Output layer (no activation, manual softmax later)
        self.fc2 = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )


    def call(self, x, training=False):

        """
        Combining the sub-layers of the classifier

        INPUT:
        - [x]: input tensor, shape = (batch_size, time_steps, d_model)
        - [training]: whether it is training mode (used to control dropout behavior)

        OUTPUT:
        - [probs]: classification probability, shape = (batch_size, num_classes)
        """

        # Mean pooling (averaging the time steps)
        pooled = tf.reduce_mean(x, axis=1)

        # ----- Multi-layer MLP classification head -----

        # The first fully connected layer
        x = self.fc1(pooled)                    # shape = (batch, hidden_dim)

        # Using Dropout to prevent overfitting
        x = self.dropout(x, training=training)

        # The second fully connected layer to map to the number of categories and get logits (not activated)
        logits = self.fc2(x)                    # shape = (batch, num_classes)

        # Use softmax to convert logits to probability distribution
        # (the sum of each row = 1, indicating the probability of belonging to each category)
        probs = tf.nn.softmax(logits, axis=-1)

        return probs


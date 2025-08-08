"""

File Properties:
* Project Name : DL Individual Project
* File Name : audio_classifier.py
* Programmer : Yu Du
* Last Update : May 4, 2025

========================================

Functions:
* AudioClassifier::__init__ -- Initialize the classifier (output layer)
* AudioClassifier::call -- Combining the sub-layers of the classifier

"""


import tensorflow as tf


class AudioClassifier(tf.keras.layers.Layer):

    def __init__(self, num_classes=10):

        """
        Initialize the classifier (output layer)

        INPUT:
        - [num_classes]: the number of categories for classification (10 categories for the GTZAN dataset)
        """

        # Initialize the parent class tf.keras.layers.Layer
        super().__init__()

        # Fully connected layer: maps features to category space (without activation, softmax is added separately later)
        self.fc = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )


    def call(self, x):

        """
        Combining the sub-layers of the classifier

        INPUT:
        - [x]: input tensor, shape = (batch_size, time_steps, d_model)

        OUTPUT:
        - [probs]: classification probability, shape = (batch_size, num_classes)
        """

        # Mean pooling for the time dimension
        pooled = tf.reduce_mean(x, axis=1)

        # Use a fully connected layer to map to the number of categories and get logits (not activated)
        logits = self.fc(pooled)

        # Use softmax to convert logits to probability distribution
        # (the sum of each row = 1, indicating the probability of belonging to each category)
        probs = tf.nn.softmax(logits, axis=-1)

        return probs
    
    
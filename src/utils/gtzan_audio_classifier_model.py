"""

File Properties:
* Project Name : DL Individual Project
* File Name : gtzan_audio_classifier_model.py
* Programmer : Yu Du
* Last Update : May 4, 2025

========================================

Functions:
* GtzanAudioClassifierModel::__init__ -- Initialize the various module layers required for the GTZAN audio classification process
* GtzanAudioClassifierModel::call -- Assembling the layers of the GTZAN audio classification pipeline

"""


import tensorflow as tf

from models import pre_encoder
from models import encoder
from models import audio_classifier
from models import multi_layer_audio_classifier


class GtzanAudioClassifierModel(tf.keras.Model):

    def __init__(self,  num_encoder=6, **kwargs):

        """
        Initialize the various module layers required for the GTZAN audio classification process
        """

        # Initialize the parent class tf.keras.Model
        super(GtzanAudioClassifierModel, self).__init__(**kwargs)

        # Module initialization:
        # Feature pre-processing layer (two layers of convolution + GELu)
        self.obj_pre_encoder_layer = pre_encoder.PreEncoder()

        # Main encoder layer (multi-head attention + FFN stacking),
        # and here are assembled according to the specified number of encoders
        self.obj_encoder_layer = [
            encoder.Encoder() for _ in range(num_encoder)
        ]

        # Output classifier layer (MeanPooling + Linear + Softmax)
        self.obj_audio_classifier_layer = audio_classifier.AudioClassifier()

        # Output classifier layer (MeanPooling + Feed-Forward * 2 + Softmax)
        # self.obj_audio_classifier_layer = multi_layer_audio_classifier.AudioClassifier()


    def call(self, inputs,training=False, return_class=False):

        """
        Assembling the layers of the GTZAN audio classification pipeline

        INPUT:
        - [inputs]: input audio features, shape = (batch_size, time_steps, channels)
        - [num_encoder_layers]: number of layers of encoder connection
        - [training]: whether it is training mode (used to control Dropout and output behaviors)

        OUTPUT:
        - [output]: classification probability distribution, shape = (batch_size, num_classes)
          or the category index (column number) corresponding to the maximum probability, shape = (batch, 1)
        """

        # Perform feature pre-processing layer (two layers of convolution)
        x = self.obj_pre_encoder_layer(inputs)

        # Loop through the main encoder layer stack (multi-head attention + FFN stack)
        for encoder_block in self.obj_encoder_layer:
            x = encoder_block(x, training=training)

        # Execute the output classifier layer
        probs = self.obj_audio_classifier_layer(x)

        # Determine the output based on whether it is for the training process
        if return_class:

            # If it is not a training process, the category index (column number)
            # corresponding to the maximum probability is taken as the output
            predicted_class = tf.argmax(probs, axis=-1, output_type=tf.int32)

            # Convert to column vector (batch, 1)
            return tf.expand_dims(predicted_class, axis=-1)

        else:

            # If it is a training process, the output classification probability distribution
            return probs


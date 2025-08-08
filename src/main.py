"""

File Properties:
* Project Name : DL Individual Project
* File Name : main.py
* Programmer : Yu Du
* Last Update : May 4, 2025

========================================

Functions:
* main -- The program start from here, and here we implement the complete model of GTZAN audio classification
  (PreEncoder → Encoder → AudioClassifier) and back propagation

"""


import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path                # For obtaining the current system location during testing

from data import wav_dataset_loader
from data import audio_preprocessor
from utils import gtzan_audio_classifier_model
from utils import data_processor_model


def main(timestamp=None):

    """
    The program start from here, and here we implement the complete model of GTZAN audio classification (PreEncoder → Encoder → AudioClassifier) ​​and back propagation

    INPUT:
    - [timestamp]: if a timestamp is provided, the classification program is executed directly, otherwise it is executed from the beginning of the reading
    """

    # Get the current path of the project in the system
    current_file_path = Path(__file__).resolve()

    # Get the root path of the project in the system
    root_dir = current_file_path.parent.parent

    # Define a output location for saving the model
    persistent_location = root_dir / "data" / "processed"

    # Define a persistent location for saving the data
    output_location = root_dir / "outputs"


    # Use timestamps to decide whether to start execution from getting the original data or from the processed data (loading data persistence)
    if timestamp:

        # The timestamp is not empty, and the data is loaded from the persistent data file
        x_train, x_test, y_train, y_test = audio_preprocessor.AudioPreprocessor.load_data_from_persistence(persistent_location, timestamp)

    else:

        # Otherwise, assemble the original data file path to prepare to load the original data
        raw_data_location = root_dir / "data" / "raw" / "GTZAN Dataset - Music Genre Classification" / "genres_original"

        # Initialize the data loader object and use it to load the raw data from assembled data file path
        obj_wdl = wav_dataset_loader.WavDatasetLoader(raw_data_location)
        audio_matrix, label_array, sample_rate_list, length_list, label_to_index = obj_wdl.load_data()

        # Display descriptive information of some original data
        print("\nSampling rate of the first 10 audio data:\n", np.array(sample_rate_list, dtype=np.int32)[:10], "\n")
        print("Length of the first 10 audio data:\n", length_list[:10])
        print("** Note: the length here refers to the length of the audio byte stream converted to float32 format in TensorFlow\n")
        print("All label-index mappings:\n", label_to_index, "\n")

        # Initialize the pre-processor object and use it to pre-process the raw data
        obj_ap = audio_preprocessor.AudioPreprocessor()
        preprocessed_data = obj_ap.preprocess_dataset(audio_matrix)
        x_train, x_test, y_train, y_test = obj_ap.split_dataset(preprocessed_data, label_array)

    # Initialize the packaged model
    obj_gacm = gtzan_audio_classifier_model.GtzanAudioClassifierModel()

    # Compile the model
    obj_gacm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),                 # Use Adam optimizer with the specified learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Set the loss function fits the multiple classification model
        metrics=['accuracy']                                                    # Set the measure standards with accuracy
    )

    # Define a timestamp for distinguishing the saved model file
    timestamp = int(time.time())

    # Set the path to automatically save the best model
    persistent_model_file_name = "best_model_" + str(timestamp) + ".weights.h5"

    # Configure callback function
    callbacks = [

        # Configure automatic learning rate adjustment
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),

        # Configure the model to terminate automatically
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True
        ),

        # Configure automatic saving of the model with the highest accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_location / persistent_model_file_name,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # Fit the model
    trained_obj_gacm = obj_gacm.fit(
        x_train,                                # Training data set
        y_train,                                # Training label set
        batch_size=16,                          # Use mini-batch strategy for training
        epochs=50,                              # Set training epochs
        validation_data=(x_test, y_test),       # Use the divided validation set
        callbacks=callbacks,                    # Configure callback function
        verbose=1                               # Set to turn on the display of training progress
    )

    # Save the trained model as a 'h5' format file
    persistent_model_file_name = "gtzan_audio_classifier_model_" + str(timestamp) + ".weights.h5"
    obj_gacm.save_weights(output_location / persistent_model_file_name)

    # Print all training loss and accuracy, and validation loss and accuracy
    print("\nTraining Loss: ", trained_obj_gacm.history["loss"])
    print("Training Accuracy: ", trained_obj_gacm.history.get("accuracy"))

    print("Validation Loss: ", trained_obj_gacm.history.get("val_loss"))
    print("Validation Accuracy: ", trained_obj_gacm.history.get("val_accuracy"), "\n")

    # Plot the training vs validation loss curves
    plt.plot(trained_obj_gacm.history['loss'], label="Training Loss")
    if "val_loss" in trained_obj_gacm.history:
        plt.plot(trained_obj_gacm.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    # Save the loss curve to local output
    loss_curve_file_name = "loss_curve_" + str(timestamp) + ".png"
    plt.savefig(output_location / loss_curve_file_name)

    plt.show()
    plt.close()


# Execution
if __name__ == "__main__":

    # Test suite execution using persistent data
    # obj_data_processor_model = data_processor_model.DataProcessorModel("test")
    # timestamp = obj_data_processor_model.persistent_data()
    # main(timestamp)

    # The complete data set is executed using persistent data
    # obj_data_processor_model = data_processor_model.DataProcessorModel("train")
    # timestamp = obj_data_processor_model.persistent_data()
    # main(timestamp)

    # The full data set is executed without using persistent data
    main()

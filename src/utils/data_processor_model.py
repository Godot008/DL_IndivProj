"""

File Properties:
* Project Name : DL Individual Project
* File Name : data_processor_model.py
* Programmer : Yu Du
* Last Update : May 4, 2025

========================================

Functions:
* DataProcessorModel::__init__ -- Used to initialize data file path and data loading and data preprocessing classes
* DataProcessorModel::get_data_variables -- Get the pre-processed data set and data description information
* DataProcessorModel::persistent_data --

"""


from pathlib import Path
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import wav_dataset_loader
from data import audio_preprocessor


class DataProcessorModel():

    def __init__(self, data_usage="test"):

        """
        Used to initialize data file path and data loading and data preprocessing classes

        INPUT:
        - [data_usage]: purpose of data usage
        """

        # Get the current path of the system
        current_file_path = Path(__file__).resolve()
        self._root_dir = current_file_path.parent.parent.parent

        # Select the data path to use based on data usage
        if data_usage == "train":
            data_path = self._root_dir / "data" / "raw" / "GTZAN Dataset - Music Genre Classification" / "genres_original"
        else:
            data_path = self._root_dir / "data" / "test" / "GTZAN Dataset - Music Genre Classification" / "genres_original"

        # Initialize data loading and data preprocessing classes
        self.obj_wav_data_loader = wav_dataset_loader.WavDatasetLoader(data_path)
        self.obj_audio_preprocessor = audio_preprocessor.AudioPreprocessor()


    def get_data_variables(self):

        """
        Get the pre-processed data set and data description information
        """

        # Get the original data set and data description information
        audio_matrix, label_array, sample_rate_list, length_list, label_to_index = self.obj_wav_data_loader.load_data()

        # Preprocess the data and divide it into training and test sets
        preprocessed_data = self.obj_audio_preprocessor.preprocess_dataset(audio_matrix)
        x_train, x_test, y_train, y_test = self.obj_audio_preprocessor.split_dataset(preprocessed_data, label_array)

        return x_train, x_test, y_train, y_test, sample_rate_list, length_list, label_to_index


    def persistent_data(self, output_dir=None):

        """
        Loading and preprocessing datasets in a persistent manner

        INPUT:
        - [output_dir]: Specify the location where persistent data is saved
          (if not provided, the directory location will be set by default)
        """

        # Determine whether to specify the data persistence location
        if not output_dir:
            output_dir = self._root_dir / "data" / "processed"

        # Load and save the original dataset to a persistent location
        timestamp = self.obj_wav_data_loader.load_data_with_persistence(output_dir)
        audio_matrix, label_array = self.obj_wav_data_loader.load_data_from_persistence(output_dir, timestamp)

        # Preprocess and partition the original data set and save it to a persistent location
        preprocessed_data = self.obj_audio_preprocessor.preprocess_dataset(audio_matrix)
        timestamp = self.obj_audio_preprocessor.split_dataset_with_persistence(preprocessed_data, label_array, output_dir)

        return timestamp


# Only used for testing this class (default comment)
# if __name__ == "__main__":
#
#     obj_data_processor_model = DataProcessorModel()
#
#     x_train, x_test, y_train, y_test, sample_rate_list, length_list, label_to_index = obj_data_processor_model.get_data_variables()
#     print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#     print(np.array(sample_rate_list, dtype=np.int32)[:3])
#     print(length_list[:3])
#     print(label_to_index)
#
#     obj_data_processor_model.persistent_data()

"""

File Properties:
* Project Name : DL Individual Project
* File Name : wav_dataset_loader.py
* Programmer : Yu Du
* Last Update : April 16, 2025

========================================

Functions:
* WavDatasetLoader::__init__ -- Initialize wav dataset loader parameters
* WavDatasetLoader::_gather_files -- Gather all wav file paths and their labels
* WavDatasetLoader::_load_wav -- Load the single wav file, and get their waveform data with float32 format,
  sampling rate and total sampling number (handling with single channel by default)
* WavDatasetLoader::_determine_max_length -- Find the maximum sampling number
* WavDatasetLoader::load_data -- Load data from all wav files as a NumPy format matrix
* WavDatasetLoader::load_data_with_persistence -- In addition to the 'load_data()' method,
  another layer of persistence operation is encapsulated
* WavDatasetLoader::load_data_from_persistence -- Load data from the specified data persistence location
  according to the specified timestamp

"""


import os                   # For manipulating files and folders
import time

import numpy as np
import tensorflow as tf

# from pathlib import Path    # For obtaining the current system location during testing


class WavDatasetLoader:

    def __init__(self, root_dir):

        """
        Initialize wav dataset loader parameters

        INPUTS:
        - [root_dir]: root directory of the raw data
        """

        self.root_dir = root_dir        # Root directory of the raw data

        self.label_to_index = {}        # For storing the mappings from folder names to numeric labels
        self.file_paths = []            # For storing all wav file paths
        self.labels = []                # For storing all corresponding labels (folder names)
        self._gather_files()            # Gather the file paths and labels

        self.desired_samples = None     # Desired samples (automatically obtain later)
        self._determine_max_length()    # Find the maximum audio length

    
    def _gather_files(self):

        """
        Gather all wav file paths and their labels
        """

        # Get all sub-directory names as label names
        label_names = [
            name for name in sorted(os.listdir(self.root_dir))
            if os.path.isdir(os.path.join(self.root_dir, name)) and not name.startswith('.')
        ]

        # Set mappings from label names to indexes (numeric labels)
        self.label_to_index = {name: idx for idx, name in enumerate(label_names)}

        # Concatenate file paths
        for label in label_names:

            # Get each folder paths according to label names
            folder = os.path.join(self.root_dir, label)

            # Skip items that are not folders
            if not os.path.isdir(folder):
                continue

            # Traverse wav files in each folder
            for file in os.listdir(folder):

                # Only process wav format files
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(folder, file))
                    self.labels.append(label)
                else:
                    pass


    def _load_wav(self, file_path):

        """
        Load the single wav file, and get their waveform data with float32 format,
        sampling rate and total sampling number (handling with single channel by default)

        INPUT:
        - [file_path]: the wav file path

        OUTPUT:
        - [audio]: waveform data
        - [sample_rate]: sample rate of the single wav file
        - [length]: total sampling number of the single wav file

        NOTE:
        GTZAN Dataset - Music Genre Classification/genres_original/jazz/jazz.00054.wav
        (the audio file cannot be read)
        """

        try:

            # Get binary byte stream from path wav file
            audio_binary = tf.io.read_file(file_path)

            # Extract audio content from byte stream (and automatically convert to float32 format in TensorFlow)
            audio, sample_rate = tf.audio.decode_wav(audio_binary)

            # Get total number of sampling points
            length = tf.shape(audio)[0].numpy()

            # Remove the last dimension i.e. (samples, 1) -> (samples, )
            audio = tf.squeeze(audio, axis=-1)

            return audio, sample_rate, length
        
        except Exception as e:
            print(f"\nThere is an error in the file and it cannot be read:\n{file_path}\n")
            print(f"The reasons for the error are as follows:\n{e}\n")
            return None, None, 0
    

    def _determine_max_length(self):

        """
        Find the maximum sampling number
        """

        # Initialize a logger
        max_len = 0

        # Traverse all wav files for finding maximum sampling number
        for file_path in self.file_paths:
            _, _, length = self._load_wav(file_path)
            max_len = max(max_len, length)

        # Set desired samples
        self.desired_samples = max_len
    

    def load_data(self):

        """
        Load data from all wav files as a NumPy format matrix

        OUTPUT:
        - [audio_matrix]: all wav file audio data
        - [label_array]: all wav file label data
        - [sample_rate_list]: sampling rate information for all wav files
        - [length_list]: length information for all wav files
        - [label_to_index]: all mappings from labels to indexes
        """

        # Initialize temporary storage lists
        audio_list = []         # For storing all wav file audio data
        label_list = []         # For storing all wav file label data
        sample_rate_list = []   # For storing all wav file sample rate information
        length_list = []        # For storing all wav file length information

        # Traverse all wav files
        for file_path, label in zip(self.file_paths, self.labels):

            # Extract data from the wav file
            audio, sample_rate, length = self._load_wav(file_path)

            # Skip unusual audio
            if audio is None:
                continue

            # Zero padding or truncation to target length
            if length < self.desired_samples:
                padding = self.desired_samples - length
                audio = tf.pad(audio, [[0, padding]])
            else:
                audio = audio[:self.desired_samples]

            # Add information
            audio_list.append(audio.numpy())
            label_list.append(self.label_to_index[label])
            sample_rate_list.append(sample_rate)
            length_list.append(length)

        # Convert to NumPy array
        audio_matrix = np.stack(audio_list).astype(np.float32)
        label_array = np.array(label_list, dtype=np.int32)

        return audio_matrix, label_array, sample_rate_list, length_list, self.label_to_index


    def load_data_with_persistence(self, output_dir):

        """
        In addition to the 'load_data()' method, another layer of persistence operation is encapsulated

        INPUT:
        - [output_dir]: persistently saved location

        OUTPUT:
        - [timestamp]: the timestamp of the persistent data file
        """

        # Execute data loading normally
        audio_matrix, label_array, sample_rate_list, length_list, _ = self.load_data()

        # Define a timestamp
        timestamp = int(time.time())

        # Save audio_matrix
        with open(os.path.join(output_dir, "audio_matrix_" + str(timestamp) + ".txt"), "w") as f:
            for i, row in enumerate(audio_matrix):
                row_str = ", ".join(str(sample) for sample in row)
                f.write(f"{i}:{row_str}\n")         # Use ":" to separate the sequence number and data

        # Save label_array
        with open(os.path.join(output_dir, "label_array_" + str(timestamp) + ".txt"), "w") as f:
            for i, label in enumerate(label_array):
                f.write(f"{i}:{label}\n")           # Use ":" to separate the sequence number and data

        # Save sample_rate_list and length_list
        with open(os.path.join(output_dir, "sample_rate_&_length_" + str(timestamp) + ".txt"), "w") as f:
            for i, (sr, length) in enumerate(zip(sample_rate_list, length_list)):
                f.write(f"{i}:{sr}, {length}\n")    # Use ":" to separate the sequence number and data

        # Save label_to_index
        with open(os.path.join(output_dir, "label_to_index_" + str(timestamp) + ".txt"), "w") as f:
            for label, index in self.label_to_index.items():
                f.write(f"{label}: {index}\n")
        
        return timestamp

    
    @staticmethod
    def load_data_from_persistence(persistent_location, timestamp):

        """
        Load data from the specified data persistence location according to the specified timestamp

        INPUT:
        - [persistent_file_path]: the location of the persistent data files
        - [timestamp]: the timestamp of the persistent data file to read

        OUTPUT:
        - [audio_matrix]: all wav file audio data
        - [label_array]: all wav file label data
        """

        # Initialize temporary result storage lists
        audio_list = []
        label_list = []

        # Concatenate persistent file paths
        audio_file_path = os.path.join(persistent_location, "audio_matrix_" + str(timestamp) + ".txt")
        label_file_path = os.path.join(persistent_location, "label_array_" + str(timestamp) + ".txt")

        # Read and parse audio_matrix data
        with open(audio_file_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue

                # Separate the sequence number and data
                _, data_str = line.split(":")

                # Parsing data and store it in the cache list
                samples = [float(x) for x in data_str.split(",")]
                audio_list.append(samples[1:])
        
        # Read and parse label_array data
        with open(label_file_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue

                # Separate the sequence number and label data
                _, label = line.split(":")

                # Store the label data in the cache list
                label_list.append(int(label))

        # Convert them to NumPy arrays
        audio_matrix = np.array(audio_list, dtype=np.float32)
        label_array = np.array(label_list, dtype=np.int32)

        return audio_matrix, label_array
    

# Only used for testing this class (default comment)
# if __name__ == "__main__":

#     current_file_path = Path(__file__).resolve()
#     root_dir = current_file_path.parent.parent.parent

#     test_data_path = root_dir / "data" / "test" / "GTZAN Dataset - Music Genre Classification" / "genres_original"
#     # raw_data_path = root_dir / "data" / "raw" / "GTZAN Dataset - Music Genre Classification" / "genres_original"
#     obj_wav_data_loader = WavDatasetLoader(test_data_path)
#     # obj_wav_data_loader = WavDatasetLoader(raw_data_path)

#     persistent_location = root_dir / "outputs"
#     timestamp = obj_wav_data_loader.load_data_with_persistence(output_dir=persistent_location)
#     audio_matrix, label_array = obj_wav_data_loader.load_data_from_persistence(persistent_location=persistent_location, timestamp=timestamp)
#     print("audio_matrix:")
#     print(audio_matrix[:10])
#     print("label_array")
#     print(label_array[:10])

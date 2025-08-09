"""

File Properties:
* Project Name : DL Individual Project
* File Name : audio_preprocessor.py
* Programmer : Yu Du
* Last Update : April 21, 2025

========================================

Functions:
* AudioPreprocessor::__init__ -- Initialize audio preprocessor parameters
* AudioPreprocessor::waveform_to_log_mel -- Convert a single audio waveform to a log-Mel spectrum
* AudioPreprocessor::normalize -- Normalize NumPy matrix to mean=0, std=1
* AudioPreprocessor::spectrogram_augment -- Perform simple data augmentation on the log-Mel spectrogram,
  including time masking and frequency masking
* AudioPreprocessor::preprocess_dataset -- Perform log-Mel feature extraction on all audio waveform data
  and stacked into a three-dimensional array: number of samples * number of frames * number of mel channels
* AudioPreprocessor::split_dataset -- Divide the preprocessed data into training set and test set
* AudioPreprocessor::split_dataset_with_persistence -- In addition to the 'split_dataset()' method,
  another layer of persistence operation is encapsulated
* AudioPreprocessor::load_data_from_persistence -- Load the flattened log-Mel features
  from the specified data persistence location according to the specified timestamp
  and restore them to the structure of (number of samples, number of frames, number of mel channels)

"""


import os
import time

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split        # For dividing training set and test set

# from pathlib import Path                                  # For obtaining the current system location during testing
# import wav_dataset_loader                                 # For obtaining original persistent data


class AudioPreprocessor:

    def __init__(self, sample_rate=22050, frame_length=1024, frame_step=256, num_mel_bins=128, time_mask_width=30, freq_mask_width=15):

        """
        Initialize audio preprocessor parameters

        INPUT:
        - [sample_rate]: sampling rate, in Hz
        - [frame_length]: number of sampling points per frame
        - [frame_step]: frame shift (number of samples skipped)
        - [num_mel_bins]: number of Mel filters
        """

        self.sample_rate = sample_rate          # Audio sampling rate
        self.frame_length = frame_length        # Frame length (sampling points)
        self.frame_step = frame_step            # Frame Shift
        self.num_mel_bins = num_mel_bins        # Number of Mel filters

        self.time_mask_width = time_mask_width  # Time dimension occlusion width
        self.freq_mask_width = freq_mask_width  # Frequency dimension occlusion width

    
    def waveform_to_log_mel(self, waveform):

        """
        Convert a single audio waveform to a log-Mel spectrum

        INPUT:
        - [waveform]: 1D Tensor, raw waveform in float32 format

        OUTPUT:
        - [log_mel_spectrogram]: 2D Tensor, Log-Mel spectrum, shape = (number of frames, number of mel channels)
        """

        # Perform STFT (short-time Fourier transform) to obtain the complex spectrum
        stft = tf.signal.stft(
            waveform, 
            frame_length=self.frame_length, 
            frame_step=self.frame_step, 
            fft_length=self.frame_length
        )

        # Get the spectrum modulus (amplitude spectrum)
        spectrogram = tf.abs(stft)

        # Create a transformation matrix from linear spectrum to Mel spectrum
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,                 # Number of mel channels
            num_spectrogram_bins=spectrogram.shape[-1],     # Number of spectral bands (usually equal to frame_length // 2 + 1)
            sample_rate=self.sample_rate, 
            lower_edge_hertz=80.0,                          # Minimum frequency
            upper_edge_hertz=self.sample_rate / 2.0         # Maximum frequency (Nyquist frequency)
        )

        # Use matrix transformation to convert linear spectrum to mel spectrum
        mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

        # Add a very small value to avoid log(0) going to negative infinity
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

        return log_mel_spectrogram
    

    def normalize(self, matrix):

        """
        Normalize NumPy matrix to mean=0, std=1

        INPUT:
        - [matrix]: input matrix

        OUTPUT:
        - [norm_matrix]: the normalized matrix
        """

        mean_val = np.mean(matrix)
        std_val = np.std(matrix)

        # Avoid dividing by 0 (normalize to 0 when all values ​​are the same)
        if std_val == 0:
            return np.zeros_like(matrix)
        
        return (matrix - mean_val) / std_val


    def spectrogram_augment(self, log_mel_spectrogram):

        """
        Perform simple data augmentation on the log-Mel spectrogram, including time masking and frequency masking

        INPUT:
        - [matrix]: log-Mel spectrum matrix

        OUTPUT:
        - [augmented_matrix]: enhanced log-mel spectrogram (shape unchanged)
        """

        # Get the time dimension (horizontal axis) and frequency dimension (vertical axis) of the log-mel spectrogram
        time_steps = tf.shape(log_mel_spectrogram)[0]                       # Timeframes
        mel_bins = tf.shape(log_mel_spectrogram)[1]                         # Number of Mel channels

        # ----- Time Occlusion -----

        # Randomly generate an occlusion starting point t0, occlusion starts from t0
        t0 = tf.random.uniform([], 0, time_steps - self.time_mask_width, dtype=tf.int32)

        # Construct an occlusion mask: dimension (time_steps, mel_bins)
        # The first t0 rows are kept (value is 1), the middle time_mask_width rows are set to 0 (blocked),
        # and the remaining rows are set to 1
        time_mask = tf.concat([
            tf.ones([t0, mel_bins]),                                        # The previous part remains unchanged
            tf.zeros([self.time_mask_width, mel_bins]),                     # Occlusion segment is set to 0
            tf.ones([time_steps - t0 - self.time_mask_width, mel_bins])     # The latter part remains unchanged
        ], axis=0)

        # ----- Frequency Blocking -----

        # Randomly select the starting point f0 of the frequency mask (column index on the Mel axis)
        f0 = tf.random.uniform([], 0, mel_bins - self.freq_mask_width, dtype=tf.int32)

        # Construct a frequency mask mask: dimension (time_steps, mel_bins)
        # In each row, the range of f0~f0+freq_mask_width is 0, and the rest are 1
        freq_mask = tf.concat([
            tf.ones([time_steps, f0]),                                      # Left side unchanged
            tf.zeros([time_steps, self.freq_mask_width]),                   # Intermediate Occlusion
            tf.ones([time_steps, mel_bins - f0 - self.freq_mask_width])     # The right side remains unchanged
        ], axis=1)

        # Multiply the masks of time masking and frequency masking to form a joint mask
        # Then do element-by-element multiplication on the original log-mel spectrum,
        # and the blocked area will become 0
        return log_mel_spectrogram * time_mask * freq_mask
    

    def preprocess_dataset(self, audio_matrix):

        """
        Perform log-Mel feature extraction on all audio waveform data and stacked
        into a three-dimensional array: number of samples * number of frames * number of mel channels

        INPUT:
        - [audio_matrix]: NumPy array, all wav file audio data,
          shape = (number of samples, number of sampling points)

        OUTPUT:
        - [log_mel_list]: NumPy array, all audio spectrum feature data after log-Mel conversion,
          shape = (number of samples, number of frames, number of mel channels)
        """

        # Initialize a storage for saving all audio spectrum feature data
        log_mel_list = []

        # Traverse all the original audio data for conversion
        for waveform in audio_matrix:

            # Convert a NumPy waveform to a TensorFlow tensor
            waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

            # Perform log-mel conversion
            log_mel = self.waveform_to_log_mel(waveform_tensor)

            # Normalize the data
            norm_log_mel = self.normalize(log_mel.numpy())

            # Data augmentation
            augmented_log_mel = self.spectrogram_augment(norm_log_mel)

            # Append to list
            log_mel_list.append(augmented_log_mel)
        
        return np.array(log_mel_list)
    

    def split_dataset(self, data, labels, test_size=0.2, random_state=2025):

        """
        Divide the preprocessed data into training set and test set

        INPUT:
        - [data]: NumPy array, preprocessed dataset
        - [labels]: NumPy array, label array
        - [test_size]: test set ratio
        - [random_state]: random seed to ensure repeatability

        OUTPUT:
        - [x_train]: preprocessed training set
        - [x_test]: preprocessed testing set
        - [y_train]: preprocessed training labels
        - [y_test]: preprocessed testing labels
        """

        # Use sklearn's tools to partition the data (while keeping the label distribution consistent)
        return train_test_split(data, labels, test_size=test_size, random_state=random_state, stratify=labels)
    

    def split_dataset_with_persistence(self, data, labels, output_dir, test_size=0.2, random_state=2025):

        """
        In addition to the 'split_dataset()' method, another layer of persistence operation is encapsulated

        INPUT:
        - [data]: NumPy array, preprocessed dataset
        - [labels]: NumPy array, label array
        - [output_dir]: persistently saved location
        - [test_size]: test set ratio
        - [random_state]: random seed to ensure repeatability

        OUTPUT:
        - [timestamp]: the timestamp of the persistent data file
        """

        # Execute data splitting normally
        x_train, x_test, y_train, y_test = self.split_dataset(data, labels, test_size=test_size, random_state=random_state)

        # Define a timestamp
        timestamp = int(time.time())

        # Save x_train
        with open(os.path.join(output_dir, "x_train_" + str(timestamp) + ".txt"), "w") as f:
            for i, sample in enumerate(x_train):
                flat = sample.flatten()                             # Flatten into a 1D array
                data_str = ",".join(str(x) for x in flat)
                f.write(f"{i}-{sample.shape[0]}:{data_str}\n")      # Use ":" to separate the data header (sequence number-frame number) and data

        # Save x_test
        with open(os.path.join(output_dir, "x_test_" + str(timestamp) + ".txt"), "w") as f:
            for i, sample in enumerate(x_test):
                flat = sample.flatten()                             # Flatten into a 1D array
                data_str = ",".join(str(x) for x in flat)
                f.write(f"{i}-{sample.shape[0]}:{data_str}\n")      # Use ":" to separate the data header (sequence number-frame number) and data

        # Save y_train
        with open(os.path.join(output_dir, "y_train_" + str(timestamp) + ".txt"), "w") as f:
            for i, label in enumerate(y_train):
                f.write(f"{i}:{label}\n")                           # Use ":" to separate the sequence number and data

        # Save y_test
        with open(os.path.join(output_dir, "y_test_" + str(timestamp) + ".txt"), "w") as f:
            for i, label in enumerate(y_test):
                f.write(f"{i}:{label}\n")                           # Use ":" to separate the sequence number and data

        return timestamp


    @staticmethod
    def load_data_from_persistence(persistent_location, timestamp, mel_bins=128):

        """
        Load the flattened log-Mel features from the specified data persistence location
        according to the specified timestamp and restore them to the structure of
        (number of samples, number of frames, number of mel channels)

        INPUT:
        - [persistent_file_path]: the location of the persistent data files
        - [timestamp]: the timestamp of the persistent data file to read
        - [mel_bins]: used to restore the flattened log-Mel feature data to the correct number of Mel channels

        OUTPUT:
        - [x_train]: preprocessed training set
        - [x_test]: preprocessed testing set
        - [y_train]: preprocessed training labels
        - [y_test]: preprocessed testing labels
        """

        # Initialize temporary result storage lists
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        # Concatenate persistent file paths
        x_train_path = os.path.join(persistent_location, "x_train_" + str(timestamp) + ".txt")
        x_test_path = os.path.join(persistent_location, "x_test_" + str(timestamp) + ".txt")
        y_train_path = os.path.join(persistent_location, "y_train_" + str(timestamp) + ".txt")
        y_test_path = os.path.join(persistent_location, "y_test_" + str(timestamp) + ".txt")

        # Read and parse x_train data
        with open(x_train_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue
                
                # Separate header and data
                header, data_str = line.split(":", 1)

                # Parsing the header
                _, frame_str = header.split("-")
                frame_count = int(frame_str)

                # Restore the data structure and store it in the cache list
                data_flat = [float(x) for x in data_str.split(",") if x]
                feature = np.array(data_flat).reshape((frame_count, mel_bins))
                x_train_list.append(feature)

        # Read and parse x_test data
        with open(x_test_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue
                
                # Separate header and data
                header, data_str = line.split(":", 1)

                # Parsing the header
                _, frame_str = header.split("-")
                frame_count = int(frame_str)

                # Restore the data structure and store it in the cache list
                data_flat = [float(x) for x in data_str.split(",") if x]
                feature = np.array(data_flat).reshape((frame_count, mel_bins))
                x_test_list.append(feature)

        # Read and parse y_train data
        with open(y_train_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue

                # Separate header and data
                _, label = line.split(":")

                # Store the data in the cache list
                y_train_list.append(int(label))

        # Read and parse y_test data
        with open(y_test_path, "r") as f:
            for line in f:

                line = line.strip()

                # Skip empty line
                if not line:
                    continue

                # Separate header and data
                _, label = line.split(":")

                # Store the data in the cache list
                y_test_list.append(int(label))

        # Restore them to NumPy arrays respectively
        x_train = np.array(x_train_list, dtype=np.float32)
        x_test = np.array(x_test_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int32)
        y_test = np.array(y_test_list, dtype=np.int32)

        return x_train, x_test, y_train, y_test

    
# Only used for testing this class (default comment)
# if __name__ == "__main__":
#
#     current_file_path = Path(__file__).resolve()
#     root_dir = current_file_path.parent.parent.parent
#
#     persistent_location = root_dir / "outputs"
#     # timestamp = 1743858266      # For testing
#     timestamp = 1743719545      # For production
#     audio_matrix, label_array = wav_dataset_loader.WavDatasetLoader.load_data_from_persistence(persistent_location, timestamp)
#
#     obj_audio_preprocessor = AudioPreprocessor()
#     preprocessed_data = obj_audio_preprocessor.preprocess_dataset(audio_matrix)
#     timestamp = obj_audio_preprocessor.split_dataset_with_persistence(preprocessed_data, label_array, persistent_location)
#     x_train, x_test, y_train, y_test = obj_audio_preprocessor.load_data_from_persistence(persistent_location, timestamp)
#
#     timestamp = 1744304161 # For testing
#     timestamp = 1744304479 # For production
#     x_train, x_test, y_train, y_test = AudioPreprocessor.load_data_from_persistence(persistent_location, timestamp)
#     print("x_train:\n", x_train.shape)
#     print("y_train:\n", y_train.shape)
#     print("x_test:\n", x_test.shape)
#     print("y_test:\n", y_test.shape)

import tensorflow as tf
from pathlib import Path

class DataLoader:

    def __init__(self, dir_path):

        self.dir_path = Path(dir_path)

    
    def load_single_wav(self, file_path):

        # Get binary byte stream from path wav file
        audio_binary = tf.io.read_file(file_path)

        # Extract audio content from byte stream (and automatically convert to float32 format in TensorFlow) and sampling rate
        audio, sample_rate = tf.audio.decode_wav(audio_binary)
        
        return audio, sample_rate
    
    
    def load_single_wav_with_label(self, file_path, label):
        
        # Get binary byte stream from path wav file
        audio_binary = tf.io.read_file(file_path)

        # Extract audio content from byte stream (and automatically convert to float32 format in TensorFlow)
        audio, _ = tf.audio.decode_wav(audio_binary)

        return audio, label

    
    def load_multiple_wavs(self):

        # Concatenate the complete file path
        pattern = str(self.dir_path / "*.wav")

        # Get the full path names of all wav files from the specified directory
        wav_dataset = tf.data.Dataset.list_files(pattern)

        return wav_dataset.map(lambda x: self.load_single_wav(x))
    

    def load_multiple_folders(self):

        # Get all subfolder paths
        subfolders = [sf for sf in self.dir_path.iterdir() if sf.is_dir()]

        datasets = []

        for idx, folder in enumerate(subfolders):

            label = idx
            pattern = str(folder / "*.wav")

            wav_files = tf.data.Dataset.list_files(pattern)

            labeled_dataset = wav_files.map(lambda x: self.load_single_wav_with_label(x, label))

            datasets.append(labeled_dataset)

        final_dataset = datasets[0]

        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds)

        return final_dataset


# Only used for testing this class (default comment)
# if __name__ == "__main__":

#     current_file = Path(__file__).resolve()
#     root_dir = current_file.parent.parent.parent
#     data_path = root_dir / "data" / "test" / "GTZAN Dataset - Music Genre Classification" / "genres_original" / "blues"
#     obj_data_loader = DataLoader(data_path)

#     raw_data = obj_data_loader.load_multiple_wavs()
#     for audio, sr in raw_data.take(1):
#         print(audio.shape, sr.numpy())
#         print(audio[0:10,])
    
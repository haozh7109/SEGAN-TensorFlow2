# -*- coding: utf-8 -*-
"""
Created on Mon. Jan 3 10:54:03 2022

@author: zhaoh
"""

from pathlib import Path
from Utility.Py_data_Preprocess_Hao import WaveformDataGenerator


# define the base path
base_path = Path('/SSD/training_set_canteenFB/Data_for_SEGAN')

# define the path to the source datasets
clean_train_folder = base_path / 'trainset_clean'
noisy_train_folder = base_path / 'trainset_noisy'
clean_test_folder  = base_path / 'testset_clean'
noisy_test_folder  = base_path / 'testset_noisy'

# define the path to save datasets after preprocessing
serialized_train_folder = base_path / 'preproc/serialized_train_data'
serialized_test_folder  = base_path / 'preproc/serialized_test_data'

# create directories if they don't exist
serialized_train_folder.mkdir(parents=True, exist_ok=True)
serialized_test_folder.mkdir(parents=True, exist_ok=True)

# define the parameters for preprocessing
sample_rate = 16000    # data sample rate (Hz)
window_size = 2 ** 14  # about 1 second of samples
stride      = 0.5      # 50% overlap between adjacent windows

def main():
    # Generate the input and target data (waveform) for training based on defined window size and stride
    Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)

    # Generate the training and test datasets
    for dataset_type in ['train', 'test']:
        Waveform_generator.process_and_serialize(dataset_type, clean_train_folder, noisy_train_folder, serialized_train_folder, clean_test_folder, noisy_test_folder, serialized_test_folder)
    print('Complete data pre-processing...')

if __name__ == '__main__':
    main()
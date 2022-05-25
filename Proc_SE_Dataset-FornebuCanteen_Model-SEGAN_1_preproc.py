# -*- coding: utf-8 -*-
"""
Created on Mon. Jan 3 10:54:03 2022

@author: zhaoh
"""

import os
from Py_data_Preprocess_Hao import WaveformDataGenerator

# define the path to the source datasets
clean_train_folder = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/trainset_clean'
noisy_train_folder = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/trainset_noisy'
clean_test_folder  = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/testset_clean'
noisy_test_folder  = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/testset_noisy'

# define the path to save datasets after preprocessing
serialized_train_folder = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/preproc/serialized_train_data'
serialized_test_folder  = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/preproc/serialized_test_data'
if(not os.path.exists(serialized_train_folder)):
    os.makedirs(serialized_train_folder)
if(not os.path.exists(serialized_test_folder)):
    os.makedirs(serialized_test_folder)


# define the parameters for preprocessing
sample_rate = 16000    # data sample rate (Hz)
window_size = 2 ** 14  # about 1 second of samples
stride      = 0.5      # 50% overlap between adjacent windows


if __name__ == '__main__':

    # Generate the input and target data (waveform) for training based on defined window size and stride
    Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)

    # Generate the training and test datasets
    Waveform_generator.process_and_serialize('train', clean_train_folder,noisy_train_folder,serialized_train_folder,clean_test_folder,noisy_test_folder,serialized_test_folder)
    Waveform_generator.process_and_serialize('test', clean_train_folder,noisy_train_folder,serialized_train_folder,clean_test_folder,noisy_test_folder,serialized_test_folder)
    print(' complete data pre-processing...')

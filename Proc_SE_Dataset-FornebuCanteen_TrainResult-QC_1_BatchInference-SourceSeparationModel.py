# -*- coding: utf-8 -*-
"""
Created on Friday, March 11 10:36:03 2022
@author: zhaoh
"""

import glob
import os
import time
import multiprocessing

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy

import pandas as pd
from numpy import savetxt
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Py_data_Preprocess_Hao import SpeechSetStatistics, speech_set_statistics_calculation
from Py_customized_denoising_models import model_generator
from Py_plot_hao import plot1d, plot2d

###############################################################################
#  (0) run GPU version of model on CPU only to utilize the multi-cpus
###############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###############################################################################
#  (1) set parameters for spectrogram generation and data loading
###############################################################################
n_fft                = 128
hop_length           = 32
sample_rate          = 16000
segment_len          = 8
train_epochs         = 200
training_flag        = False
inference_batch_size = 100

###############################################################################
#  (2) get the dataset for training
###############################################################################
# set parent Directory path
parent_dir = "/TensorFlowSSD/training_set_canteenFornebu/"

# set data directory
NoisyData_directory   = "noisy/"
CleanData_directory   = "clean/"
DenoiseData_directory = "denoised/"
QC_directory          = "QC/"
stats_info_directory  = "denoised/SourceSeparation/"

# set data paths
NoisyData_path   = os.path.join(parent_dir, NoisyData_directory)
CleanData_path   = os.path.join(parent_dir, CleanData_directory)
DenoiseData_path = os.path.join(parent_dir, DenoiseData_directory)
QC_path          = os.path.join(parent_dir, QC_directory)
stats_info_path  = os.path.join(parent_dir, stats_info_directory)

# get the filename list
filenames_NoisyData = sorted(glob.glob(os.path.join(NoisyData_path, '*.wav')))
filenames_CleanData = sorted(glob.glob(os.path.join(CleanData_path, '*.wav')))

filenames_NoisyData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx, filename in enumerate(filenames_NoisyData)])
filenames_CleanData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx, filename in enumerate(filenames_CleanData)])

filenames_NoisyData_list = [filename for id, filename in filenames_NoisyData_sorted]
filenames_CleanData_list = [filename for id, filename in filenames_CleanData_sorted]

# split data list into train/validation/test sets
train_NoisyData_list, test_NoisyData_list, train_CleanData_list, test_CleanData_list   = train_test_split(filenames_NoisyData_list, filenames_CleanData_list, test_size=0.1, random_state=42)
train_NoisyData_list, valid_NoisyData_list, train_CleanData_list, valid_CleanData_list = train_test_split(train_NoisyData_list, train_CleanData_list, test_size=0.1, random_state=42)

# apply statistical analysis on the all data sets ( load the calculated statistics from the file if it exists)
stats_train_NoisyData, stats_valid_NoisyData, stats_test_NoisyData, stats_train_CleanData, stats_valid_CleanData, stats_test_CleanData = speech_set_statistics_calculation(train_NoisyData_list, valid_NoisyData_list, test_NoisyData_list, train_CleanData_list, valid_CleanData_list, test_CleanData_list, verbose=True, Data_path=stats_info_path)

###############################################################################
#  (3) Generate the QC files
###############################################################################
# # load the previous trained model for QC generation
QC_selected_models     = ['SE_Dataset-FornebuCanteen_Model-Baseline_200epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline2Frames_200epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline_AugmentTime_100epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline_AugmentTimeSpec_200epochs', 'SE_Dataset-FornebuCanteen_Model-Test_Architect_200epochs', 'SE_Dataset-FornebuCanteen_Model-SourceSeparation_200epochs']
QC_selected_model      = QC_selected_models[-1]

if QC_selected_model == QC_selected_models[0]:
    model_name = 'Baseline'
elif QC_selected_model == QC_selected_models[1]:
    model_name = 'Baseline2Frames'
    segment_len = 2
elif QC_selected_model == QC_selected_models[2]:
    model_name = 'Baseline_AugmentTime'
elif QC_selected_model == QC_selected_models[3]:
    model_name = 'Baseline_AugmentTimeSpec'
elif QC_selected_model == QC_selected_models[4]:
    model_name = 'Test_Architect'
else:
    model_name = 'SourceSeparation'

# define the denoising-inference method, which will be used for multiple-cpu execution
def denoise_model_inference_func(job_id):

    # import tensorflow inside the function which requires by the multiple-cpu setting in python
    import tensorflow as tf

    # load the model
    NN_model_loaded = tf.keras.models.load_model(QC_selected_model, compile=False)
    NN_model_loaded.summary()

    # select qc noisy file
    QC_NoisyFile_lists = test_NoisyData_list[:inference_batch_size]
    noisy_file         = QC_NoisyFile_lists[job_id]

    # generate the QC file
    if not os.path.exists(os.path.join(QC_path + 'Lock/', 'lock_process_snr_fileID_' + str(job_id) + '_.lck')):

        # generate the lock key
        with open(os.path.join(QC_path + 'Lock/', 'lock_process_snr_fileID_' + str(job_id) + '_.lck'), 'w') as fp:
            pass

        #load noisy data
        noisy_signal, sample_rate_NoisyData = librosa.load(noisy_file, sr=sample_rate)
        clean_signal, sample_rate_CleanData = librosa.load(CleanData_path + '/clean_fileid' + noisy_file.split('/')[-1].split('fileid')[-1], sr=sample_rate)

        # generate the spectrogram of noisy input
        noisy_stft = librosa.core.stft(noisy_signal, hop_length=hop_length, n_fft=n_fft, window=scipy.signal.windows.hann(128))
        noisy_spec = np.abs(noisy_stft)

        # transform to log domain
        noisy_spec = librosa.amplitude_to_db(noisy_spec, top_db=200.0)

        # make the prediction based on the trained neural net
        denoised_spec = np.zeros(noisy_spec.shape)

        for segment_index in np.arange(segment_len - 1, noisy_spec.shape[1]):

            # display the job status
            print("------ job is running on the segments {} of {} --------\n".format(segment_index, noisy_spec.shape[1]))

            # extract the vector elements for the data
            x_element = noisy_spec[:, segment_index - segment_len + 1: segment_index + 1]

            # apply normalization
            x_element = (x_element - stats_test_NoisyData['mean']) / stats_test_NoisyData['std']

            # flatten to 1D array
            x_element = x_element.flatten()

            # make the prediction
            mask_predicted = np.squeeze(NN_model_loaded.predict(np.expand_dims(x_element, axis=0)),axis=0)

            # unflatten the predicted mask
            mask_predicted = np.reshape(mask_predicted, (n_fft // 2 + 1, segment_len))

            # apply the predicted mask to the noisy spectrogram (apply in linear scaled spectrogram domain)
            y_element = librosa.db_to_amplitude(noisy_spec[:, segment_index - segment_len + 1: segment_index + 1]) * mask_predicted

            # output to the spectrogram
            # denoised_spec[:, segment_index] = np.reshape(y_element, (denoised_spec.shape[0]))
            denoised_spec[:, segment_index] = np.reshape(y_element[:, -1], [-1])

        # Remove artifacts generated by the heading None-values in the spectrogram by filling the min value in each freq component
        denoised_spec_MeanRow = np.reshape(denoised_spec[:, segment_len - 1:].min(axis=1), [denoised_spec.shape[0], 1])
        denoised_spec[:, :segment_len - 1] = np.tile(denoised_spec_MeanRow, [1, segment_len - 1])

        # transform back to time domain
        # denoised_spec = librosa.db_to_amplitude(denoised_spec)

        # generate the phase of spectrogram from the noisy data
        noisy_stft = librosa.core.stft(noisy_signal, hop_length=hop_length, n_fft=n_fft, window=scipy.signal.windows.hann(128))
        phase_noisy_stft = np.angle(noisy_stft)
        mag_denoisy_stft = denoised_spec

        # reconstruct the denoised  signal in time domain
        denoised_signal_stft = mag_denoisy_stft * np.cos(phase_noisy_stft) + mag_denoisy_stft * np.sin(phase_noisy_stft) * 1j
        denoised_signal = librosa.core.istft(denoised_signal_stft, window=scipy.signal.windows.hann(128), length=noisy_signal.__len__())  # use length parameter to pad the output in order to keep same length with the input

        #  save the output as wav files
        sf.write(QC_path + 'Clean_' + noisy_file.split('/')[-1], clean_signal, sample_rate)
        sf.write(QC_path + 'Noisy_' + noisy_file.split('/')[-1], noisy_signal, sample_rate)
        sf.write(QC_path + 'QC_Model-' + model_name + '_NN_Denoised_' + noisy_file.split('/')[-1], denoised_signal, sample_rate)

    else:
        print("-- Denoising process has applied already --")

###############################################################################
#  (4) Generate the QC files
###############################################################################

if __name__ ==  '__main__':

    print('-----------------------------------------------------------')
    print('----------Start denoising : ----------')
    print('-----------------------------------------------------------')

    start = time.time()

    ## run single inference for test
    # denoise_model_inference_func(10)

    ## set up the multiprocess based job-execution
    # (1) option-1 : use process
    processes = []

    for job_id in np.arange(0, inference_batch_size):
        P = multiprocessing.Process(target=denoise_model_inference_func, args=(job_id,))
        processes.append(P)
        P.start()

    for process in processes:
        process.join()

    # # (2) option-2 : use pool
    # pool = Pool()
    # pool.map(denoise_model_inference_func, range(0, inference_batch_size))
    # pool.close()

    running_time = time.time() - start
    print('-----------------------------------------------------------')
    print("Denoising completed in : {:0.2f}s".format(running_time))
    print('-----------------------------------------------------------')

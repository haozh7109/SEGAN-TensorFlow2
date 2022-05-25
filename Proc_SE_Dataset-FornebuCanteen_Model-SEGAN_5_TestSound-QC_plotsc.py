# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:54:03 2020

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


# -----------------------------------------------------------------------------
#  (1) define global variables and predefined methods
# -----------------------------------------------------------------------------
# set parent Directory path
parent_dir = "/home/hao_listen/PycharmProjects/pythonProject/Code_reference/SEGAN-master2/SEGAN-master/"

# set data directory
NoisyData_directory    = "data/output_data"
CleanData_directory    = "data/output_data"
DenoisedData_directory = "data/output_data"
QC_Data_directory      = "data/QC_data"

# generate the folders
NoisyData_path    = os.path.join(parent_dir, NoisyData_directory)
CleanData_path    = os.path.join(parent_dir, CleanData_directory)
DenoisedData_path = os.path.join(parent_dir, DenoisedData_directory)
QC_Data_path      = os.path.join(parent_dir, QC_Data_directory)

try:
    os.makedirs(QC_Data_path)
except OSError as error:
    print(error)

# get the filename list
filenames_NoisyData = sorted(glob.glob(os.path.join(NoisyData_path, 'noisy*.wav')))

# save the processed used filename list
savetxt(QC_Data_path + '/All_Noisy_File_list.txt', filenames_NoisyData, fmt='%s')

# -----------------------------------------------------------------------------
#  (2) generate the QC plots
# -----------------------------------------------------------------------------

for i_speech in np.arange(filenames_NoisyData.__len__()):

    # print the status
    print('Processing the %d-th speech file...' % (i_speech + 1))

    # get the fileID
    filename = filenames_NoisyData[i_speech].split("/")[-1].split('.wav')[0].split('noisy_')[-1]

    # get the noisy file name
    noisy_filename = filenames_NoisyData[i_speech]
    noisy_signal, sr_noisy = librosa.load(noisy_filename, 16000)
    
    # get the denoised files' name
    denoised_filename = DenoisedData_path + '/denoised_' + filename + '.wav'
    denoise_signal, sr_denoise_signal = librosa.load(denoised_filename, 16000)

    # get the clean file name
    clean_filename = CleanData_path + '/clean_' + filename + '.wav'
    clean_signal, sr_clean_signal = librosa.load(clean_filename, 16000)

    # -----------------------------------------------------------------------------
    #  (3) generate the QC Spectra plots
    # -----------------------------------------------------------------------------
    # define the parameters
    sample_rate = sr_clean_signal
    hop_length  = 256
    n_fft       = 1024

    clean_stft = librosa.core.stft(clean_signal, hop_length=hop_length, n_fft=n_fft, window=scipy.signal.windows.hann(n_fft))
    noisy_stft = librosa.core.stft(noisy_signal, hop_length=hop_length, n_fft=n_fft, window=scipy.signal.windows.hann(n_fft))
    denoised_stft = librosa.core.stft(denoise_signal, hop_length=hop_length, n_fft=n_fft, window=scipy.signal.windows.hann(n_fft))

    clean_spec    = np.abs(clean_stft)
    noisy_spec    = np.abs(noisy_stft)
    denoised_spec = np.abs(denoised_stft)

    clean_spec    = librosa.amplitude_to_db(clean_spec, top_db=200)
    noisy_spec    = librosa.amplitude_to_db(noisy_spec, top_db=200)
    denoised_spec = librosa.amplitude_to_db(denoised_spec, top_db=200)

    # plot the denoised spectrogram
    plt.figure(figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.db_to_amplitude(noisy_spec), sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of noisy audio')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.db_to_amplitude(clean_spec), sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of clean audio')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.db_to_amplitude(denoised_spec), sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of denoised audio')
    plt.colorbar()
    plt.show()
    plt.savefig(QC_Data_path + '/' + 'QC_Figure1_' + filename + '.png')

    # plot the denoised spectrogram
    plt.figure(figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.subplot(3, 1, 1)
    librosa.display.specshow(noisy_spec, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of noisy audio (Log scale)')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(clean_spec, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of clean audio (Log scale)')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(denoised_spec, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency (logrithm)")
    plt.title('Spectrogram of denoised audio (Log scale)')
    plt.colorbar()
    plt.savefig(QC_Data_path + '/' + 'QC_Figure2_' + filename + '.png')

    # -----------------------------------------------------------------------------
    #  (2.1) generate the QC waveform plots
    # -----------------------------------------------------------------------------

    plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    plt.subplot(3,1,1)
    librosa.display.waveplot(noisy_signal,sr=sample_rate)
    plt.xlabel('')
    plt.ylim([-1,1])
    plt.title("Synthetic noisy signal")
    plt.show()

    plt.subplot(3,1,2)
    librosa.display.waveplot(clean_signal,sr=sample_rate)
    plt.ylim([-1,1])
    plt.xlabel('')
    plt.title("Original clean signal")

    plt.subplot(3,1,3)
    librosa.display.waveplot(denoise_signal,sr=sample_rate)
    plt.ylim([-1,1])
    plt.title("Convolutional Neural Network derived denoised signal")
    plt.show()
    plt.savefig(QC_Data_path + '/' + 'QC_Figure3_' + filename + '.png')


    # close the figures before the next iteration
    plt.close('all')


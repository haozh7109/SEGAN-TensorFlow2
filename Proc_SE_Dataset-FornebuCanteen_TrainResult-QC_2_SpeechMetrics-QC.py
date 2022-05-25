# -*- coding: utf-8 -*-
"""
Created on Thursday， March 24 13:54:03 2022

@author: zhaoh
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
from numpy import savetxt, loadtxt, savez_compressed, load
from pesq import pesq
from pystoi import stoi
import sounddevice as sd
import torch
from torchmetrics import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio, MeanSquaredError
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

###############################################################################
#  (0) set parameters for the dataset
###############################################################################
n_fft = 128
hop_length = 32
sample_rate = 16000
segment_len = 8
model_names  = ['Baseline', 'Baseline2Frames', 'Baseline_AugmentTimeSpec', 'Baseline_AugmentTimeSpec-NoiseDamping10dB','SourceSeparation']
model_titles = ['Baseline', 'Baseline2Frames',  'Baseline_AugT+S', 'Data:SNR+10dB', 'SpeechSeparation']
N_models = len(model_names)

# -----------------------------------------------------------------------------
#  (1) generate the file lists
# -----------------------------------------------------------------------------

# set parent Directory path
parent_dir = "/TensorFlowSSD/training_set_canteenFornebu/"
QC_directory = "QC/"

# generate the folders
QC_path = os.path.join(parent_dir, QC_directory)

try:
    os.makedirs(QC_path)
except OSError as error:
    print(error)

# get the filename list
filenames_NoisyData = sorted(glob.glob(os.path.join(QC_path, 'Noisy*.wav')))

# save the processed used filename list
savetxt(QC_path + '/All_Noisy_File_list.txt', filenames_NoisyData, fmt='%s')

# -----------------------------------------------------------------------------
#  (2) calculate the metrics for the dataset
# -----------------------------------------------------------------------------

pesq_scores  = np.zeros((filenames_NoisyData.__len__(), N_models + 1))
SNR_scores   = np.zeros((filenames_NoisyData.__len__(), N_models + 1))
SISDR_scores = np.zeros((filenames_NoisyData.__len__(), N_models + 1))
MSE_scores   = np.zeros((filenames_NoisyData.__len__(), N_models + 1))
STOI_scores  = np.zeros((filenames_NoisyData.__len__(), N_models + 1))


pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb')
snr = SignalNoiseRatio()
si_sdr = ScaleInvariantSignalDistortionRatio()
mse = MeanSquaredError()
stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)

for index_model in np.arange(0, N_models):

    # get the model name
    model_name = model_names[index_model]

    # caculate the metrics for the each data sample
    for i_speech in np.arange(filenames_NoisyData.__len__()):
        # print the job progress
        print('----------calculating the metrics for the ' + str(i_speech + 1) + 'th speech sample, model: ' + model_name + '----------')

        # get clean, noisy and denoised speech
        noisy_filename = filenames_NoisyData[i_speech]
        file_id = noisy_filename.split('_')[-1].split('.wav')[0]
        clean_filename = glob.glob(os.path.join(QC_path, f'Clean*{file_id}*'))[0]
        denoised_filename = glob.glob(os.path.join(QC_path, f'QC_Model-{model_name}_NN_Denoised*{file_id}*'))[0]

        # load the sound files
        ref, sr_input = librosa.load(clean_filename, sr=sample_rate)
        noisy, sr_input = librosa.load(noisy_filename, sr=sample_rate)
        denoised_method, sr_input = librosa.load(denoised_filename, sr=sample_rate)

        # calculate pesq values (https://torchmetrics.readthedocs.io/en/latest/references/modules.html#perceptualevaluationspeechquality)
        pesq_noisy = pesq(torch.tensor(ref), torch.tensor(noisy))
        pesq_denoised = pesq(torch.tensor(ref), torch.tensor(denoised_method))

        # calculate SNR values
        snr_noisy = snr(torch.tensor(noisy), torch.tensor(ref))
        snr_denoised = snr(torch.tensor(denoised_method), torch.tensor(ref))

        # calculate SI-SDR values
        sisdr_noisy = si_sdr(torch.tensor(noisy), torch.tensor(ref))
        sisdr_denoised = si_sdr(torch.tensor(denoised_method), torch.tensor(ref))

        # calculate MSE values
        mse_noisy = np.sum(np.square(noisy - ref)) / len(ref)
        mse_denoised = np.sum(np.square(denoised_method - ref)) / len(ref)

        # calculate STOI values
        stoi_noisy = stoi(torch.tensor(noisy), torch.tensor(ref))
        stoi_denoised = stoi(torch.tensor(denoised_method), torch.tensor(ref))

        # save into metrics
        pesq_scores[i_speech, index_model] = pesq_denoised
        SNR_scores[i_speech, index_model] = snr_denoised
        SISDR_scores[i_speech, index_model] = sisdr_denoised
        MSE_scores[i_speech, index_model] = mse_denoised
        STOI_scores[i_speech, index_model] = stoi_denoised
        pesq_scores[i_speech, -1] = pesq_noisy
        SNR_scores[i_speech, -1] = snr_noisy
        SISDR_scores[i_speech, -1] = sisdr_noisy
        MSE_scores[i_speech, -1] = mse_noisy
        STOI_scores[i_speech, -1] = stoi_noisy

# -----------------------------------------------------------------------------
#  (3) making the QC plots
# -----------------------------------------------------------------------------

# generate the sorting index
index_pesq = pesq_scores[:, -1].argsort()

# sorting the metrics according to the pesq scores
pesq_scores_sorted = pesq_scores[index_pesq]
SNR_scores_sorted = SNR_scores[index_pesq]
SISDR_scores_sorted = SISDR_scores[index_pesq]
MSE_scores_sorted = MSE_scores[index_pesq]
STOI_scores_sorted = STOI_scores[index_pesq]

# save the metrics into csv files
np.savetxt(os.path.join(QC_path, 'pesq_scores.csv'), pesq_scores, delimiter=',')
np.savetxt(os.path.join(QC_path, 'SNR_scores.csv'), SNR_scores, delimiter=',')
np.savetxt(os.path.join(QC_path, 'SISDR_scores.csv'), SISDR_scores, delimiter=',')
np.savetxt(os.path.join(QC_path, 'MSE_scores.csv'), MSE_scores, delimiter=',')
np.savetxt(os.path.join(QC_path, 'STOI_scores.csv'), STOI_scores, delimiter=',')

# select the portion of data for the metrics calculation (if it is needed)
useful_selection = 65
pesq_scores_sorted = pesq_scores_sorted[:useful_selection,:]
SNR_scores_sorted = SNR_scores_sorted[:useful_selection,:]
SISDR_scores_sorted = SISDR_scores_sorted[:useful_selection,:]
MSE_scores_sorted = MSE_scores_sorted[:useful_selection,:]
STOI_scores_sorted = STOI_scores_sorted[:useful_selection,:]

# (1) plot the individual metrics
def metrics_plot(metrics, title, xlabel, ylabel, data_labels, filename):
    plt.figure(figsize=(12, 8))
    for i in np.arange(0, len(data_labels)):
        plt.plot(metrics[:, i], label=data_labels[i])
    plt.legend()
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


metrics_plot(pesq_scores_sorted, 'PESQ scores', 'Speech sample index', 'PESQ score', model_names + ['Noisy data'], os.path.join(QC_path, 'QC_PESQ_scores.png'))
metrics_plot(SNR_scores_sorted, 'SNR scores', 'Speech sample index', 'SNR score', model_names + ['Noisy data'], os.path.join(QC_path, 'QC_SNR_scores.png'))
metrics_plot(SISDR_scores_sorted, 'SI-SDR scores', 'Speech sample index', 'SI-SDR score', model_names + ['Noisy data'], os.path.join(QC_path, 'QC_SISDR_scores.png'))
metrics_plot(MSE_scores_sorted, 'MSE scores', 'Speech sample index', 'MSE score', model_names + ['Noisy data'], os.path.join(QC_path, 'QC_MSE_scores.png'))
metrics_plot(STOI_scores_sorted, 'STOI scores', 'Speech sample index', 'STOI score', model_names + ['Noisy data'], os.path.join(QC_path, 'QC_STOI_scores.png'))


# (2) bar plot the metrics mean
def bar_plot(metrics, title, xlabel, ylabel, data_labels, filename):
    fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), dpi=100)
    rects = plt.bar(np.arange(0, len(data_labels)), metrics, tick_label=data_labels)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)

    # add values on the bar plot
    for p in rects:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')

    plt.savefig(filename, bbox_inches='tight')
    plt.show()


bar_plot(np.mean(pesq_scores_sorted[:, [-1] + list(range(pesq_scores_sorted.shape[1] - 1))], axis=0), 'PESQ scores mean', 'Model', 'PESQ score', ['Noisy data'] + model_titles, os.path.join(QC_path, 'QC_PESQ_scores_mean.png'))
bar_plot(np.mean(SNR_scores_sorted[:, [-1] + list(range(SNR_scores_sorted.shape[1] - 1))], axis=0), 'SNR scores mean', 'Model', 'SNR score', ['Noisy data'] + model_titles, os.path.join(QC_path, 'QC_SNR_scores_mean.png'))
bar_plot(np.mean(SISDR_scores_sorted[:, [-1] + list(range(SISDR_scores_sorted.shape[1] - 1))], axis=0), 'SI-SDR scores mean', 'Model', 'SI-SDR score', ['Noisy data'] + model_titles, os.path.join(QC_path, 'QC_SISDR_scores_mean.png'))
bar_plot(np.mean(MSE_scores_sorted[:, [-1] + list(range(MSE_scores_sorted.shape[1] - 1))], axis=0), 'MSE scores mean', 'Model', 'MSE score', ['Noisy data'] + model_titles, os.path.join(QC_path, 'QC_MSE_scores_mean.png'))
bar_plot(np.mean(STOI_scores_sorted[:, [-1] + list(range(STOI_scores_sorted.shape[1] - 1))], axis=0), 'STOI scores mean', 'Model', 'STOI score', ['Noisy data'] + model_titles, os.path.join(QC_path, 'QC_STOI_scores_mean.png'))

print('=============== QC job completed ===============')

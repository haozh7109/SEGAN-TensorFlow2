# -*- coding: utf-8 -*-
"""
Created on Thur Oct 18 21:54:03 2020

@author: zhaoh
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from tqdm import tqdm
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter


# ------------------------------------------------------------------------------
# (1) Class for Spectrogram processing
# ------------------------------------------------------------------------------

class SpectrogramProcess(object):

    def __init__(self, sample_rate=None, n_fft=None, hop_length=None, input_segment_len=None, target_segment_len=None, top_db=200.0, vec_spec_output=True):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_segment_len = input_segment_len
        self.target_segment_len = target_segment_len
        self.top_db = top_db
        self.vec_spec_output = vec_spec_output

    # method to generate the magnitude based spectrogram
    def generate_spec(self, noisy_filenames_array, clean_filenames_array=None, frame_increment=1):

        # get total file size
        file_size = noisy_filenames_array.__len__()

        # generate the spectrogram for training and target
        x = []
        y = []

        for file_index in np.arange(file_size):

            # load noisy signal
            NoisyData, sample_rate_NoisyData = librosa.load(noisy_filenames_array[file_index], sr=self.sample_rate)

            # load clean signal
            if clean_filenames_array is None:
                filename_CleanData = noisy_filenames_array[file_index].split('noisy')[0] + 'clean/' + 'clean_fileid_' + noisy_filenames_array[file_index].split('/')[-1].split('_')[-1]
            else:
                filename_CleanData = clean_filenames_array[file_index]

            CleanData, sample_rate_CleanData = librosa.load(filename_CleanData, sr=self.sample_rate)

            # generate the spectrogram for clean and noisy data
            clean_stft = librosa.core.stft(CleanData, hop_length=self.hop_length, n_fft=self.n_fft, window=scipy.signal.windows.hann(self.n_fft))  # This setting has been updated in the data preprocessing !!
            clean_spec = np.abs(clean_stft)

            noisy_stft = librosa.core.stft(NoisyData, hop_length=self.hop_length, n_fft=self.n_fft, window=scipy.signal.windows.hann(self.n_fft))  # This setting has been updated in the data preprocessing !!
            noisy_spec = np.abs(noisy_stft)

            # transform spectrogram to dB scale
            clean_spec = librosa.amplitude_to_db(clean_spec, top_db=self.top_db)
            noisy_spec = librosa.amplitude_to_db(noisy_spec, top_db=self.top_db)

            # Example: generate input (x) with dimension Freq * input_segment_len   , and target (y) with dimension Freq * target_segment_len

            for segment_index in np.arange(self.input_segment_len - 1, clean_spec.shape[1], frame_increment):
                # extract the vector elements for the data
                x_element = noisy_spec[:, segment_index - self.input_segment_len + 1: segment_index + 1]

                if self.vec_spec_output:
                    y_element = clean_spec[:, segment_index]
                else:
                    y_element = clean_spec[:, segment_index - self.target_segment_len + 1: segment_index + 1]

                # save the data into the list
                x.append(x_element)
                y.append(y_element)

        # convert data back to the array
        input_data = np.array(x)
        target_data = np.array(y)

        # output the generated input and target
        return input_data, target_data

    # method to generate the complex spectrogram
    def generate_complex_spec(self, noisy_filenames_array, clean_filenames_array=None, frame_increment=1):

        # get total file size
        file_size = noisy_filenames_array.__len__()

        # generate the spectrogram for training and target
        x = []
        y = []

        for file_index in np.arange(file_size):

            # load noisy signal
            NoisyData, sample_rate_NoisyData = librosa.load(noisy_filenames_array[file_index], sr=self.sample_rate)

            # load clean signal
            if clean_filenames_array is None:
                filename_CleanData = noisy_filenames_array[file_index].split('noisy')[0] + 'clean/' + 'clean_fileid_' + noisy_filenames_array[file_index].split('/')[-1].split('_')[-1]
            else:
                filename_CleanData = clean_filenames_array[file_index]

            CleanData, sample_rate_CleanData = librosa.load(filename_CleanData, sr=self.sample_rate)

            # generate the spectrogram for clean and noisy data
            clean_stft = librosa.core.stft(CleanData, hop_length=self.hop_length, n_fft=self.n_fft, window=scipy.signal.windows.hann(self.n_fft))  # This setting has been updated in the data preprocessing !!
            noisy_stft = librosa.core.stft(NoisyData, hop_length=self.hop_length, n_fft=self.n_fft, window=scipy.signal.windows.hann(self.n_fft))  # This setting has been updated in the data preprocessing !!

            # save real and imaginary part of the spectrogram
            clean_stft_complex = np.stack([clean_stft.real, clean_stft.imag], axis=-1)
            noisy_stft_complex = np.stack([noisy_stft.real, noisy_stft.imag], axis=-1)

            # padding the spectrogram if the time-frame number is less than the pre-defined frame number
            if clean_stft_complex.shape[1] < self.input_segment_len:
                padding_size = self.input_segment_len - clean_stft_complex.shape[1]
                clean_stft_complex = np.pad(clean_stft_complex, ((0, 0), (0, padding_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                noisy_stft_complex = np.pad(noisy_stft_complex, ((0, 0), (0, padding_size), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

            # generate the input and the target
            for segment_index in np.arange(self.input_segment_len - 1, clean_stft_complex.shape[1], frame_increment):

                # extract the vector elements for the data
                x_element = noisy_stft_complex[:, segment_index - self.input_segment_len + 1: segment_index + 1, :]

                if self.vec_spec_output:
                    y_element = clean_stft_complex[:, segment_index, :]
                else:
                    y_element = clean_stft_complex[:, segment_index - self.target_segment_len + 1: segment_index + 1, :]

                # save the data into the list
                x.append(x_element)
                y.append(y_element)

        # convert data back to the array
        input_data = np.array(x)
        target_data = np.array(y)

        # output the generated input and target
        return input_data, target_data


# ------------------------------------------------------------------------------
# (2) Class for Waveform Data processing
# ------------------------------------------------------------------------------

class WaveformDataGenerator(object):

    # constructor
    def __init__(self, sample_rate, window_size, stride):
        """
        :param sample_rate: the sample rate of the waveform data
        :param window_size: the window size of the waveform data sliding
        :param stride: the percentage of the window size for the sliding
        """
        # set the parameters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.stride = stride

    def slice_signal(self, file, window_size, stride, sample_rate, waveform_type='raw', padding=False):
        """
        function for slicing the audio file
        by window size and sample rate with [1-stride] percent overlap (default 50%).
        """
        if waveform_type == 'raw':
            wav, sr = librosa.load(file, sr=sample_rate)
        else:
            wav = file

        hop = int(window_size * stride)
        slices = []

        if padding == False:
            for end_idx in range(window_size, len(wav), hop):
                start_idx = end_idx - window_size
                slice_sig = wav[start_idx:end_idx]
                slices.append(slice_sig)
        else: # padding with zeros for the last window, to avoid the data truncation
            for end_idx in range(window_size, len(wav), hop):
                start_idx = end_idx - window_size
                slice_sig = wav[start_idx:end_idx]
                slices.append(slice_sig)

            if end_idx < len(wav):
                end_idx = end_idx + hop
                start_idx = end_idx - window_size
                slice_sig = wav[start_idx:]
                # slice_sig_padded = np.pad(slice_sig, (0, window_size - len(slice_sig)), 'constant', constant_values=(0, 0))
                slice_sig_padded = np.pad(slice_sig, (0, window_size - len(slice_sig)), 'symmetric')
                slices.append(slice_sig_padded)

        return slices

    def reconstruct_from_slice(self, slices_lists, window_size, stride, orginal_data_length=None):
        """
        function for reconsturct signal from sliced audio segments
        by window size and sample rate with [1-stride] percent overlap (default 50%).
        """

        # get the hop size
        hop = int(window_size * stride)

        # get the total frame number
        total_frame_num = len(slices_lists)

        # get the length of the signal
        data_reconstruct = np.zeros((total_frame_num - 1) * hop + window_size)
        data_reconstruct1 = np.zeros((total_frame_num - 1) * hop + window_size)

        # reconstruct the signal (method 1: using the average of the sliced signal) !! this approach is not good since it modifies the original signal's amplitude.
        for frame_index in np.arange(total_frame_num):

            if frame_index == 0:
                data_reconstruct[:window_size] = np.reshape(slices_lists[0], -1)
            else:
                data_reconstruct[hop + hop * (frame_index - 1): window_size + hop * (frame_index - 1)] = 0.5 * (
                        data_reconstruct[hop + hop * frame_index: window_size + hop * frame_index] + np.reshape(slices_lists[frame_index], -1)[:window_size - hop])
                data_reconstruct[window_size + hop * (frame_index - 1): window_size + hop * (frame_index)] = np.reshape(slices_lists[frame_index], -1)[window_size - hop:]

        # reconstruct the signal (method 2: adding the new frame to the previous frame)
        for frame_index in np.arange(total_frame_num):

            if frame_index == 0:
                data_reconstruct1[:window_size] = np.reshape(slices_lists[0], -1)
            else:
                data_reconstruct1[window_size + hop * (frame_index - 1): window_size + hop * (frame_index)] = np.reshape(slices_lists[frame_index], -1)[-hop:]

        # return the reconstructed signal
        if orginal_data_length == None:
            return data_reconstruct1
        else:
            return data_reconstruct1[:orginal_data_length]

    def process_and_serialize(self, data_type, clean_train_folder, noisy_train_folder, serialized_train_folder, clean_test_folder, noisy_test_folder, serialized_test_folder):
        """
        Serialize, down-sample the sliced signals and save on separate folder.
        """

        if data_type == 'train':
            clean_folder = clean_train_folder
            noisy_folder = noisy_train_folder
            serialized_folder = serialized_train_folder
        else:
            clean_folder = clean_test_folder
            noisy_folder = noisy_test_folder
            serialized_folder = serialized_test_folder
        if not os.path.exists(serialized_folder):
            os.makedirs(serialized_folder)

        # walk through the path, slice the audio file, and save the serialized result
        for root, dirs, files in os.walk(clean_folder):

            if len(files) == 0:
                continue

            for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
                clean_file = os.path.join(clean_folder, filename)
                noisy_file = os.path.join(noisy_folder, filename)

                # slice both clean signal and noisy signal
                clean_sliced = self.slice_signal(clean_file, self.window_size, self.stride, self.sample_rate)
                noisy_sliced = self.slice_signal(noisy_file, self.window_size, self.stride, self.sample_rate)

                # serialize - file format goes [original_file]_[slice_number].npy
                # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
                for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                    pair = np.array([slice_tuple[0], slice_tuple[1]])
                    np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=pair)


# ------------------------------------------------------------------------------
# (3) Class for Waveform analysis (fft)
# ------------------------------------------------------------------------------

class WaveformAnalysis(object):
    """
    Class for waveform analysis (fft, etc.)
    """

    def __init__(self, fs):
        """
        Initialize the parameters.
        """
        self.fs = fs

    def fft_analysis(self, signal, qc_flag=True):
        """
        Perform fft analysis on the signal.
        """
        # number of samples of the signal
        n = len(signal)

        # sampling spacing
        t = 1.0 / self.fs

        # frequency spacing
        f = np.arange(n) / (n * t)
        xf = fftfreq(n, t)[:n // 2]

        # perform fft
        fft_result = fft(signal)

        # plot the result
        if qc_flag:
            plt.figure(figsize=(12, 4))
            plt.plot(xf, np.abs(fft_result)[:n // 2])
            plt.grid()
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude')
            plt.title('FFT of the signal')
            plt.show()

        spec_mag = np.abs(fft_result)[:n // 2]
        spec_freq = xf

        return spec_mag, spec_freq


# ------------------------------------------------------------------------------
# (4) functions for pre-processing
# ------------------------------------------------------------------------------

def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays: input data has the dimension of [batch, channel, time]
        emph_coeff:   emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal batch: : output data has the dimension of [batch, channel, time]
    """

    result = np.zeros(signal_batch.shape)

    # input data needs to be the order of [batch, channel, time] for the following loop processing
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])

    return result


def emphasis_librosa(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a signal.

    Args:
        signal: signal, represented as numpy array: input data has the dimension of [channel, time]
        emph_coeff:   emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal: : output data has the dimension of [channel, time]
    """

    result = np.zeros(signal_batch.shape)

    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):

            if pre:
                """
                Pre-emphasize an audio signal with a first-order auto-regressive filter:
                y[n] -> y[n] - coef * y[n-1]
                https://librosa.org/doc/main/_modules/librosa/effects.html#preemphasis
                """
                b = np.asarray([1.0, -emph_coeff], dtype=channel_data.dtype)
                a = np.asarray([1.0], dtype=channel_data.dtype)
                zi = 2 * channel_data[..., 0] - channel_data[..., 1]
                zi = np.atleast_1d(zi)
                y_out, z_f = scipy.signal.lfilter(b, a, channel_data, zi=np.asarray(zi, dtype=channel_data.dtype))
                result[sample_idx][ch] = y_out

            else:
                """
                De-emphasize an audio signal with the inverse operation of preemphasis():
                If y = preemphasis(x, coef=coef, zi=zi), the deemphasis is:
                 >>> x[i] = y[i] + coef * x[i-1]
                 https://librosa.org/doc/main/_modules/librosa/effects.html#preemphasis
                """
                b = np.asarray([1.0, -emph_coeff], dtype=channel_data.dtype)
                a = np.asarray([1.0], dtype=channel_data.dtype)
                zi = ((2 - emph_coeff) * channel_data[0] - channel_data[1]) / (3 - emph_coeff)
                channel_data[0] -= zi
                y_out = scipy.signal.lfilter(a, b, channel_data)
                result[sample_idx][ch] = y_out

    return result


# -*- coding: utf-8 -*-
"""
Created on Tue. Jan 4 17:54:03 2022

@author: zhaoh
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Dropout, Conv1D, Conv1DTranspose, SimpleRNN, \
    AveragePooling1D, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, Softmax, concatenate, MaxPooling2D, UpSampling2D, LSTM, Reshape, LeakyReLU, Dropout
from tensorflow.keras.activations import elu, relu, tanh, sigmoid



# 1. define the customized generator model of SEGAN

class GeneratorSegan(tf.keras.Model):
    """
    class of generator model
    """

    def __init__(self, **kwargs):
        super(GeneratorSegan, self).__init__()

        # --- declare encoder layers ---

        # assume input sample size is (batch_size, sample=16384, channel=1)
        self.encoder_layer1 = Conv1D(filters=16, kernel_size=32, strides=2, padding='same',activation=None)  # output tensor shape: (batch_size, sample=8192, channel=16)
        self.encoder_layer1_act = LeakyReLU(alpha=0.25)

        self.encoder_layer2 = Conv1D(filters=32, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 4096, 32)
        self.encoder_layer2_act = LeakyReLU(alpha=0.25)

        self.encoder_layer3 = Conv1D(filters=32, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 2048, 32)
        self.encoder_layer3_act = LeakyReLU(alpha=0.25)

        self.encoder_layer4 = Conv1D(filters=64, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 1024, 64)
        self.encoder_layer4_act = LeakyReLU(alpha=0.25)

        self.encoder_layer5 = Conv1D(filters=64, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 512, 64)
        self.encoder_layer5_act = LeakyReLU(alpha=0.25)

        self.encoder_layer6 = Conv1D(filters=128, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 256, 128)
        self.encoder_layer6_act = LeakyReLU(alpha=0.25)

        self.encoder_layer7 = Conv1D(filters=128, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 128, 128)
        self.encoder_layer7_act = LeakyReLU(alpha=0.25)

        self.encoder_layer8 = Conv1D(filters=256, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 64, 256)
        self.encoder_layer8_act = LeakyReLU(alpha=0.25)

        self.encoder_layer9 = Conv1D(filters=256, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 32, 256)
        self.encoder_layer9_act = LeakyReLU(alpha=0.25)

        self.encoder_layer10 = Conv1D(filters=512, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 16, 512)
        self.encoder_layer10_act = LeakyReLU(alpha=0.25)

        self.encoder_layer11 = Conv1D(filters=1024, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 8, 1024)
        self.encoder_layer11_act = LeakyReLU(alpha=0.25)

        # --- declare decoder layers ---
        # the input tensor shape is (batch_size, 8, 2048), due to the concatenation of encoder layers' output and the input random noise tensor
        self.decoder_layer10 = Conv1DTranspose(filters=512, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 16, 512)
        self.decoder_layer10_act = LeakyReLU(alpha=0.25)

        self.decoder_layer9 = Conv1DTranspose(filters=256, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 32, 256)
        self.decoder_layer9_act = LeakyReLU(alpha=0.25)

        self.decoder_layer8 = Conv1DTranspose(filters=256, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 64, 256)
        self.decoder_layer8_act = LeakyReLU(alpha=0.25)

        self.decoder_layer7 = Conv1DTranspose(filters=128, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 128, 128)
        self.decoder_layer7_act = LeakyReLU(alpha=0.25)

        self.decoder_layer6 = Conv1DTranspose(filters=128, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 256, 128)
        self.decoder_layer6_act = LeakyReLU(alpha=0.25)

        self.decoder_layer5 = Conv1DTranspose(filters=64, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 512, 64)
        self.decoder_layer5_act = LeakyReLU(alpha=0.25)

        self.decoder_layer4 = Conv1DTranspose(filters=64, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 1024, 64)
        self.decoder_layer4_act = LeakyReLU(alpha=0.25)

        self.decoder_layer3 = Conv1DTranspose(filters=32, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 2048, 32)
        self.decoder_layer3_act = LeakyReLU(alpha=0.25)

        self.decoder_layer2 = Conv1DTranspose(filters=32, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 4096, 32)
        self.decoder_layer2_act = LeakyReLU(alpha=0.25)

        self.decoder_layer1 = Conv1DTranspose(filters=16, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 8192, 16)
        self.decoder_layer1_act = LeakyReLU(alpha=0.25)

        self.decoder_output = Conv1DTranspose(filters=1, kernel_size=32, strides=2,padding='same')  # output tensor shape: (batch_size, 16384, 1)
        self.decoder_output_act = tf.keras.layers.Activation('tanh')

    def __call__(self, x, z):
        """
        x: input noisy siginal: dim = (batch_size, 16384, 1)
        z: input latent noise vector dim = (batch_size, 8, 1024)
        """
        # --- encoder ---
        e1 = self.encoder_layer1(x)
        e1 = self.encoder_layer1_act(e1)
        e2 = self.encoder_layer2(e1)
        e2 = self.encoder_layer2_act(e2)
        e3 = self.encoder_layer3(e2)
        e3 = self.encoder_layer3_act(e3)
        e4 = self.encoder_layer4(e3)
        e4 = self.encoder_layer4_act(e4)
        e5 = self.encoder_layer5(e4)
        e5 = self.encoder_layer5_act(e5)
        e6 = self.encoder_layer6(e5)
        e6 = self.encoder_layer6_act(e6)
        e7 = self.encoder_layer7(e6)
        e7 = self.encoder_layer7_act(e7)
        e8 = self.encoder_layer8(e7)
        e8 = self.encoder_layer8_act(e8)
        e9 = self.encoder_layer9(e8)
        e9 = self.encoder_layer9_act(e9)
        e10 = self.encoder_layer10(e9)
        e10 = self.encoder_layer10_act(e10)
        e11 = self.encoder_layer11(e10)
        e11 = self.encoder_layer11_act(e11)

        # --- concatenate encoder layers' output latent vector and the input random noise vector ---
        latent = tf.concat([e11, z], axis=-1)

        # --- decoder ---
        d10 = self.decoder_layer10(latent)
        d10 = self.decoder_layer10_act(d10)
        d10 = tf.concat([d10, e10], axis=-1)

        d9 = self.decoder_layer9(d10)
        d9 = self.decoder_layer9_act(d9)
        d9 = tf.concat([d9, e9], axis=-1)

        d8 = self.decoder_layer8(d9)
        d8 = self.decoder_layer8_act(d8)
        d8 = tf.concat([d8, e8], axis=-1)

        d7 = self.decoder_layer7(d8)
        d7 = self.decoder_layer7_act(d7)
        d7 = tf.concat([d7, e7], axis=-1)

        d6 = self.decoder_layer6(d7)
        d6 = self.decoder_layer6_act(d6)
        d6 = tf.concat([d6, e6], axis=-1)

        d5 = self.decoder_layer5(d6)
        d5 = self.decoder_layer5_act(d5)
        d5 = tf.concat([d5, e5], axis=-1)

        d4 = self.decoder_layer4(d5)
        d4 = self.decoder_layer4_act(d4)
        d4 = tf.concat([d4, e4], axis=-1)

        d3 = self.decoder_layer3(d4)
        d3 = self.decoder_layer3_act(d3)
        d3 = tf.concat([d3, e3], axis=-1)

        d2 = self.decoder_layer2(d3)
        d2 = self.decoder_layer2_act(d2)
        d2 = tf.concat([d2, e2], axis=-1)

        d1 = self.decoder_layer1(d2)
        d1 = self.decoder_layer1_act(d1)
        d1 = tf.concat([d1, e1], axis=-1)

        d0 = self.decoder_output(d1)
        output = self.decoder_output_act(d0)

        return output


# 1.9 define the customized discriminator model of SEGAN
class DiscriminatorSegan(tf.keras.Model):
    """
    class of discriminator model
    """

    def __init__(self):
        super(DiscriminatorSegan, self).__init__()
        # input of discriminator: (batch_size, 16384, 2), which contains two channels: (1) a clean signal and (2) a noisy signal
        self.discriminator_layer1 = Conv1D(filters=32, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 8192, 32)
        self.vbn_layer1 = VirtualBatchNorm(32)
        self.discriminator_layer1_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer2 = Conv1D(filters=64, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 4096, 64)
        self.vbn_layer2 = VirtualBatchNorm(64)
        self.discriminator_layer2_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer3 = Conv1D(filters=64, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 2048, 64)
        self.dropout_layer1 = Dropout(rate=0.5)
        self.vbn_layer3 = VirtualBatchNorm(64)
        self.discriminator_layer3_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer4 = Conv1D(filters=128, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 1024, 128)
        self.vbn_layer4 = VirtualBatchNorm(128)
        self.discriminator_layer4_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer5 = Conv1D(filters=128, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 512, 128)
        self.vbn_layer5 = VirtualBatchNorm(128)
        self.discriminator_layer5_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer6 = Conv1D(filters=256, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 256, 256)
        self.dropout_layer2 = Dropout(rate=0.5)
        self.vbn_layer6 = VirtualBatchNorm(256)
        self.discriminator_layer6_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer7 = Conv1D(filters=256, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 128, 256)
        self.vbn_layer7 = VirtualBatchNorm(256)
        self.discriminator_layer7_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer8 = Conv1D(filters=512, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 64, 512)
        self.vbn_layer8 = VirtualBatchNorm(512)
        self.discriminator_layer8_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer9 = Conv1D(filters=512, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 32, 512)
        self.dropout_layer3 = Dropout(rate=0.5)
        self.vbn_layer9 = VirtualBatchNorm(512)
        self.discriminator_layer9_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer10 = Conv1D(filters=1024, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 16, 1024)
        self.vbn_layer10 = VirtualBatchNorm(1024)
        self.discriminator_layer10_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer11 = Conv1D(filters=2048, kernel_size=31, strides=2,padding='same')  # output tensor shape: (batch_size, 8, 2048)
        self.vbn_layer11 = VirtualBatchNorm(2048)
        self.discriminator_layer11_act = LeakyReLU(alpha=0.03)

        # post processing layer, apply 1x1 convolution for dimension reduction
        self.discriminator_layer12 = Conv1D(filters=1, kernel_size=1, strides=1,padding='same')  # output tensor shape: (batch_size, 8, 1)
        self.discriminator_layer12_act = LeakyReLU(alpha=0.03)

        self.discriminator_layer_flatten = Flatten()
        self.discriminator_layer_fully_connected = Dense(units=1)  # output tensor shape: (batch_size, 1)
        self.discriminator_layer_fully_connected_act = sigmoid

    def __call__(self, x, ref):
        """
        Args:
            x: (batch_size, 16384, 2)
            ref: (batch_size, 16384, 2)
        Returns:
            (batch_size, 1)
        """

        # step-1: only reference signal is fed into discriminator
        ref_x = self.discriminator_layer1(ref)
        ref_x, mean1, meansq1 = self.vbn_layer1(ref_x)
        ref_x = self.discriminator_layer1_act(ref_x)

        ref_x = self.discriminator_layer2(ref_x)
        ref_x, mean2, meansq2 = self.vbn_layer2(ref_x)
        ref_x = self.discriminator_layer2_act(ref_x)

        ref_x = self.discriminator_layer3(ref_x)
        ref_x = self.dropout_layer1(ref_x)
        ref_x, mean3, meansq3 = self.vbn_layer3(ref_x)
        ref_x = self.discriminator_layer3_act(ref_x)

        ref_x = self.discriminator_layer4(ref_x)
        ref_x, mean4, meansq4 = self.vbn_layer4(ref_x)
        ref_x = self.discriminator_layer4_act(ref_x)

        ref_x = self.discriminator_layer5(ref_x)
        ref_x, mean5, meansq5 = self.vbn_layer5(ref_x)
        ref_x = self.discriminator_layer5_act(ref_x)

        ref_x = self.discriminator_layer6(ref_x)
        ref_x = self.dropout_layer2(ref_x)
        ref_x, mean6, meansq6 = self.vbn_layer6(ref_x)
        ref_x = self.discriminator_layer6_act(ref_x)

        ref_x = self.discriminator_layer7(ref_x)
        ref_x, mean7, meansq7 = self.vbn_layer7(ref_x)
        ref_x = self.discriminator_layer7_act(ref_x)

        ref_x = self.discriminator_layer8(ref_x)
        ref_x, mean8, meansq8 = self.vbn_layer8(ref_x)
        ref_x = self.discriminator_layer8_act(ref_x)

        ref_x = self.discriminator_layer9(ref_x)
        ref_x = self.dropout_layer3(ref_x)
        ref_x, mean9, meansq9 = self.vbn_layer9(ref_x)
        ref_x = self.discriminator_layer9_act(ref_x)

        ref_x = self.discriminator_layer10(ref_x)
        ref_x, mean10, meansq10 = self.vbn_layer10(ref_x)
        ref_x = self.discriminator_layer10_act(ref_x)

        ref_x = self.discriminator_layer11(ref_x)
        ref_x, mean11, meansq11 = self.vbn_layer11(ref_x)

        # step-2: training signal is fed into discriminator, and uses the step-1's derived mean and meansq for batch normalization
        x = self.discriminator_layer1(x)
        x, _, _ = self.vbn_layer1(x, mean1, meansq1)
        x = self.discriminator_layer1_act(x)

        x = self.discriminator_layer2(x)
        x, _, _ = self.vbn_layer2(x, mean2, meansq2)
        x = self.discriminator_layer2_act(x)

        x = self.discriminator_layer3(x)
        x = self.dropout_layer1(x)
        x, _, _ = self.vbn_layer3(x, mean3, meansq3)
        x = self.discriminator_layer3_act(x)

        x = self.discriminator_layer4(x)
        x, _, _ = self.vbn_layer4(x, mean4, meansq4)
        x = self.discriminator_layer4_act(x)

        x = self.discriminator_layer5(x)
        x, _, _ = self.vbn_layer5(x, mean5, meansq5)
        x = self.discriminator_layer5_act(x)

        x = self.discriminator_layer6(x)
        x = self.dropout_layer2(x)
        x, _, _ = self.vbn_layer6(x, mean6, meansq6)
        x = self.discriminator_layer6_act(x)

        x = self.discriminator_layer7(x)
        x, _, _ = self.vbn_layer7(x, mean7, meansq7)
        x = self.discriminator_layer7_act(x)

        x = self.discriminator_layer8(x)
        x, _, _ = self.vbn_layer8(x, mean8, meansq8)
        x = self.discriminator_layer8_act(x)

        x = self.discriminator_layer9(x)
        x = self.dropout_layer3(x)
        x, _, _ = self.vbn_layer9(x, mean9, meansq9)
        x = self.discriminator_layer9_act(x)

        x = self.discriminator_layer10(x)
        x, _, _ = self.vbn_layer10(x, mean10, meansq10)
        x = self.discriminator_layer10_act(x)

        x = self.discriminator_layer11(x)
        x, _, _ = self.vbn_layer11(x, mean11, meansq11)
        x = self.discriminator_layer11_act(x)

        x = self.discriminator_layer12(x)
        x = self.discriminator_layer12_act(x)

        x = self.discriminator_layer_flatten(x)
        x = self.discriminator_layer_fully_connected(x)
        x = self.discriminator_layer_fully_connected_act(x)

        return x


# 1.10 define the customized virtual batch normalization layer of SEGAN
class VirtualBatchNorm(tf.Module):
    """
    class of virtual batch normalization layer; refer to the paper:https://paperswithcode.com/method/virtual-batch-normalization
    """

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super(VirtualBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # --- define gamma and beta parameters ---
        self.gamma = tf.Variable(tf.random.normal(shape=(1, 1, num_features), mean=1.0, stddev=0.02), dtype=tf.float32)
        self.beta = tf.Variable(tf.zeros(shape=(1, 1, num_features)))

    def get_stats(self, x):
        """
        get the mean and mean square of the input tensor
        :param x: input tensor
        :return: mean and mean square of the input tensor over features
        """
        mean = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
        mean_sq = tf.reduce_mean(x ** 2, axis=[0, 1], keepdims=True)
        return mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        normalize the input tensor
        :param x: input tensor
        :param mean: mean of the input tensor over features
        :param mean_sq: mean square of the input tensor over features
        :return: normalized input tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.shape) == 3  # check the input data is 3-dim

        if mean.shape[-1] != self.num_features:
            raise Exception(
                'Mean tensor size not equal to number of features : given {}, expected {}'.format(tf.shape(x)[-1], self.num_features))
        if mean_sq.shape[-1] != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'.format(tf.shape(x)[-1], self.num_features))

        # --- normalize, https://www.wikiwand.com/en/Standard_deviation ---
        std = tf.sqrt(mean_sq - mean ** 2 + self.eps)
        x = (x - mean) / std
        x = x * self.gamma + self.beta

        return x

    def __call__(self, x, ref_mean=None, ref_mean_sq=None):
        """
        call the virtual batch normalization layer
        :param x: input tensor
        :param ref_mean: reference mean over features
        :param ref_mean_sq: reference squared mean over features
        :return: normalized tensor, reference mean over features, reference squared mean over features
        """

        # --- get the mean and mean square of the input tensor ---
        mean, mean_sq = self.get_stats(x)

        if ref_mean is None or ref_mean_sq is None:
            ref_mean = mean
            ref_mean_sq = mean_sq
            output = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = tf.cast(tf.shape(x)[0],tf.float32)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            output = self.normalize(x, mean, mean_sq)

        return output, ref_mean, ref_mean_sq


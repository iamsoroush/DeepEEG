# -*- coding: utf-8 -*-
"""Model definitions.

Instantiate your desired model and fit, evaluate, and predict using that:
    import YourModel from models
    model = YourModel()
    model.create_model()
    model.compile()
    model.train()
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os

import matplotlib.pyplot as plt

from .custom_layers import InstanceNormalization, TemporalAttention, TemporalAttentionV2, TemporalAttentionV3
from . import keras

plt.style.use('ggplot')


class BaseModel:

    def __init__(self, input_shape, model_name):
        self.input_shape_ = input_shape
        self.model_name_ = model_name
        self.model_ = None
        self.loss = 'binary_crossentropy'
        self.optimizer = keras.optimizers.Adam()
        self.metrics = [keras.metrics.binary_accuracy]
        self.history = None

    def compile(self):
        self.model_.compile(loss=self.loss,
                            optimizer=self.optimizer,
                            metrics=self.metrics)


class Conv2DModel(BaseModel):

    """Lightweight 2D CNN for EEG classification.
     Source paper: Cloud-aided online EEG classification system for brain healthcare - A case study of depression
      evaluation with a lightweight CNN.
        [https://onlinelibrary.wiley.com/doi/10.1002/spe.2668]
    """

    def __init__(self, input_shape, model_name='conv2d'):
        super().__init__(input_shape, model_name)
        self.optimizer = keras.optimizers.SGD(lr=0.01,
                                              momentum=0.9,
                                              decay=1e-4,
                                              nesterov=True)
        self.loss = 'mean_squared_error'
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        x = keras.layers.Reshape((32, 32, self.input_shape_[-1]))(input_tensor)
        x = keras.layers.SpatialDropout2D(0.1)(x)
        x = keras.layers.Conv2D(20, (3, 3))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(18, (3, 3))(x)
        x = keras.layers.MaxPooling2D((1, 1))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(250, activation='sigmoid')(x)
        x = keras.layers.Dense(60, activation='sigmoid')(x)
        prediction = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input_tensor, prediction)

        self.model_ = model
        return model


class ESTCNNModel(BaseModel):

    """Spatio-temporal CNN model.

     Source paper: EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation
        [https://ieeexplore.ieee.org/document/8607897]

    """

    def __init__(self, input_shape, model_name='st_cnn'):
        super().__init__(input_shape, model_name)
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_)
        input1 = keras.layers.Permute((2, 1))(input_tensor)
        input1 = keras.layers.Lambda(keras.backend.expand_dims,
                                     arguments={'axis': -1},
                                     name='estcnn_input')(input1)

        x = self.core_block(input1, 16)
        x = keras.layers.MaxPooling2D((1, 2), strides=2)(x)

        x = self.core_block(x, 32)
        x = keras.layers.MaxPooling2D((1, 2), strides=2)(x)

        x = self.core_block(x, 64)
        x = keras.layers.AveragePooling2D((1, 7), strides=7)(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(50, activation='relu')(x)
        output_tensor = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input_tensor, output_tensor)

        self.model_ = model
        return model

    @staticmethod
    def core_block(x, n_units):
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(x)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(out)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Conv2D(filters=n_units,
                                  kernel_size=(1, 3),
                                  padding='valid',
                                  kernel_initializer='glorot_normal',
                                  activation='relu')(out)
        out = keras.layers.BatchNormalization()(out)
        return out


class BaselineDeepEEG(BaseModel):

    """SpatioTemporal 1D CNN composed of 4 SpatioTemporalConv1D layers.

    Note:
        Receptive field of each unit before GAP layer is 833 time-steps, about 3 seconds with sampling rate of 256,
        i.e. each unit before time abstraction layer looks at 3 seconds of input multi-variate time-series.
    """

    def __init__(self,
                 input_shape,
                 model_name='DeepEEG',
                 n_kernels=(6, 6, 8, 8),
                 spatial_dropout_rate=0.1,
                 dropout_rate=0.2,
                 pool_size=2,
                 pool_stride=2,
                 use_bias=False,
                 weight_norm=False,
                 attention=None,
                 normalization=None,
                 input_dropout=False):
        super().__init__(input_shape, model_name)
        self.n_kernels = n_kernels
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.spatial_dropout_rate = spatial_dropout_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.weight_norm = weight_norm
        self.attention = attention
        self.normalization = normalization
        self.input_dropout = input_dropout
        if keras.backend.image_data_format() != 'channels_last':
            keras.backend.set_image_data_format('channels_last')

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        if self.input_dropout:
            x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(input_tensor)
            x = self._spatio_temporal_conv1d(x, self.n_kernels[0] * 3, 8, 1, 1)
        else:
            x = self._spatio_temporal_conv1d(input_tensor, self.n_kernels[0] * 4, 8, 1, 1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 2
        x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(x)
        x = self._spatio_temporal_conv1d(x, self.n_kernels[1] * 3, 8, 1, 1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3 - n
        for n_units in self.n_kernels[2:]:
            x = keras.layers.Dropout(self.dropout_rate)(x)
            x = self._spatio_temporal_conv1d(x, n_units * 3, 8, 1, 1)
            x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                              strides=self.pool_stride)(x)

        # Temporal abstraction
        if self.attention is None:
            x = keras.layers.GlobalAveragePooling1D()(x)
        elif self.attention == 'v1':
            x = TemporalAttention()(x)
        elif self.attention == 'v2':
            x = TemporalAttentionV2()(x)
        else:
            x = TemporalAttentionV3()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model
        return model

    def _spatio_temporal_conv1d(self, input_tensor, filters, kernel_size, dilation_rate, strides):
        if self.weight_norm:
            norm = keras.constraints.UnitNorm(axis=(0, 1))
        else:
            norm = None
        out = keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_last',
                                  dilation_rate=dilation_rate,
                                  activation=None,
                                  use_bias=self.use_bias,
                                  kernel_constraint=norm)(input_tensor)
        if self.normalization == 'batch':
            out = keras.layers.BatchNormalization(axis=-1)(out)
        elif self.normalization == 'instance':
            out = InstanceNormalization(axis=-1, mean=0, stddev=1)(out)
        out = keras.layers.Activation('elu')(out)
        return out


class DilatedDeepEEG(BaselineDeepEEG):
    """Spatio-Temporal Dilated Filter Bank CNN.

            The design is based on DWT, i.e. each layer consists of dilated filters that extract features in different
             frequencies and different contexts.

        Note:
            Receptive field of each unit before GAP layer is 833 time-steps, about 3 seconds with sampling rate of 256,
            i.e. each unit before time abstraction layer looks at 3 seconds of input multi-variate time-series.
        """

    def __init__(self,
                 input_shape,
                 model_name='DilatedDeepEEG',
                 **kwargs):
        super().__init__(input_shape,
                         model_name,
                         **kwargs)

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        if self.input_dropout:
            x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(input_tensor)
            x = self._spatio_temporal_dilated_filter_bank(input_tensor=x,
                                                          n_units=self.n_kernels[0],
                                                          strides=1)
        else:
            x = self._spatio_temporal_dilated_filter_bank(input_tensor=input_tensor,
                                                          n_units=self.n_kernels[0],
                                                          strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 2
        x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(x)
        x = self._spatio_temporal_dilated_filter_bank(input_tensor=x,
                                                      n_units=self.n_kernels[1],
                                                      strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3 - n
        for n_units in self.n_kernels[2:]:
            x = keras.layers.Dropout(self.dropout_rate)(x)
            x = self._spatio_temporal_dilated_filter_bank(input_tensor=x,
                                                          n_units=n_units,
                                                          strides=1)
            x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                              strides=self.pool_stride)(x)

        # Temporal abstraction
        if self.attention is None:
            x = keras.layers.GlobalAveragePooling1D()(x)
        elif self.attention == 'v1':
            x = TemporalAttention()(x)
        elif self.attention == 'v2':
            x = TemporalAttentionV2()(x)
        else:
            x = TemporalAttentionV3()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model
        return model

    def _spatio_temporal_dilated_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=6,
                                                dilation_rate=1,
                                                strides=strides)
        branch_a = self._spatio_temporal_conv1d(input_tensor=branch_a,
                                                filters=n_units,
                                                kernel_size=6,
                                                dilation_rate=1,
                                                strides=strides)

        branch_b = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=8,
                                                dilation_rate=4,
                                                strides=strides)

        branch_c = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=8,
                                                dilation_rate=8,
                                                strides=strides)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output


class WindowedDeepEEG(BaselineDeepEEG):
    """Spatio-Temporal Windowed Filter Bank CNN.

            The design is based on FFT, i.e. each layer consists of  filters that extract features in different
             frequencies from the same context.

        Note:
            Receptive field of each unit before GAP layer is 833 time-steps, about 3 seconds with sampling rate of 256,
            i.e. each unit before time abstraction layer looks at 3 seconds of input multi-variate time-series.
        """

    def __init__(self,
                 input_shape,
                 model_name='WindowedDeepEEG',
                 **kwargs):
        super().__init__(input_shape,
                         model_name,
                         **kwargs)

    def create_model(self):
        input_tensor = keras.layers.Input(shape=self.input_shape_,
                                          name='input_tensor')

        # Block 1
        if self.input_dropout:
            x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(input_tensor)
            x = self._spatio_temporal_windowed_filter_bank(input_tensor=x,
                                                           n_units=self.n_kernels[0],
                                                           strides=1)
        else:
            x = self._spatio_temporal_windowed_filter_bank(input_tensor=input_tensor,
                                                           n_units=self.n_kernels[0],
                                                           strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 2
        x = keras.layers.SpatialDropout1D(self.spatial_dropout_rate)(x)
        x = self._spatio_temporal_windowed_filter_bank(input_tensor=x,
                                                       n_units=self.n_kernels[1],
                                                       strides=1)
        x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                          strides=self.pool_stride)(x)

        # Block 3 - n
        for n_units in self.n_kernels[2:]:
            x = keras.layers.Dropout(self.dropout_rate)(x)
            x = self._spatio_temporal_windowed_filter_bank(input_tensor=x,
                                                           n_units=n_units,
                                                           strides=1)
            x = keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                              strides=self.pool_stride)(x)

        # Temporal abstraction
        if self.attention is None:
            x = keras.layers.GlobalAveragePooling1D()(x)
        elif self.attention == 'v1':
            x = TemporalAttention()(x)
        elif self.attention == 'v2':
            x = TemporalAttentionV2()(x)
        else:
            x = TemporalAttentionV3()(x)

        # Logistic regression unit
        output_tensor = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(input_tensor, output_tensor)
        self.model_ = model
        return model

    def _spatio_temporal_windowed_filter_bank(self, input_tensor, n_units, strides):
        branch_a = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=64,
                                                dilation_rate=1,
                                                strides=strides)

        branch_b = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=32,
                                                dilation_rate=2,
                                                strides=strides)

        branch_c = self._spatio_temporal_conv1d(input_tensor=input_tensor,
                                                filters=n_units,
                                                kernel_size=16,
                                                dilation_rate=4,
                                                strides=strides)

        output = keras.layers.concatenate([branch_a, branch_b, branch_c], axis=-1)
        return output


def f1_score(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
    return f1_val


def sensitivity(y_true, y_pred):
    # recall: true_p / possible_p
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.backend.epsilon())


def specificity(y_true, y_pred):
    # true_n / possible_n
    true_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.backend.epsilon())
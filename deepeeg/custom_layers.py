# -*- coding: utf-8 -*-
"""Keras custom layers."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>


from . import keras


class TemporalAttention(keras.layers.Layer):
    """Attention layer for conv1d networks.

    Source paper: arXiv:1512.08756v5

    Summarizes temporal axis and outputs a vector with the same length as channels.
    Note that the unweighted attention will be simple GlobalAveragePooling1D

    Make sure to pass the inputs to this layer in "channels_last" format.

    use like this at top of a conv1d layer:
        x = TemporalAttention()(x)
    """

    def __init__(self,
                 w_regularizer=None,
                 b_regularizer=None,
                 w_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.w = None
        self.b = None
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(TemporalAttention, self).build(input_shape)

    def compute_mask(self, input_tensor, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # x: T*D , W: D*1 ==> a: T*1
        a = self._dot_product(x, self.w)

        if self.bias:
            a += self.b

        a = keras.backend.tanh(a)

        alpha = keras.backend.exp(a)
        denominator = keras.backend.sum(alpha, axis=-1, keepdims=True) + keras.backend.epsilon()
        alpha /= denominator

        # x: T*D, alpha: T*1
        weighted_input = x * keras.backend.expand_dims(alpha)
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def _dot_product(x, kernel):
        """Wrapper for dot product operation,

        Args:
            x: input
            kernel: weights
        """
        return keras.backend.squeeze(keras.backend.dot(x, keras.backend.expand_dims(kernel)), axis=-1)


class TemporalAttentionV3(keras.layers.Layer):
    """Attention layer for conv1d networks.

    Summarizes temporal axis and outputs a vector with the same length as channels.
    Note that the unweighted attention will be simple GlobalAveragePooling1D

    Make sure to pass the inputs to this layer in "channels_last" format.

    use like this at top of a conv1d layer:
        x = TemporalAttention()(x)

    Note: this version is different from TemporalAttention in normalizing weights of time steps,
        in this version weights will not normalized to be in range (0, 1). The tanh will be used
        as activation function of attention neuron.
    """

    def __init__(self,
                 w_regularizer=None,
                 b_regularizer=None,
                 w_constraint=None,
                 b_constraint=None,
                 bias=False,
                 **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.w = None
        self.b = None
        super(TemporalAttentionV3, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(TemporalAttentionV3, self).build(input_shape)

    def compute_mask(self, input_tensor, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # x: T*D , W: D*1 ==> a: T*1
        a = self._dot_product(x, self.w)

        if self.bias:
            a += self.b

        a = keras.backend.tanh(a)
        a = a / keras.backend.abs(keras.backend.sum(a, axis=-1, keepdims=True) + keras.backend.epsilon())

        # x: T*D, a: T*1
        weighted_input = x * keras.backend.expand_dims(a)
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def _dot_product(x, kernel):
        """Wrapper for dot product operation,

        Args:
            x: input
            kernel: weights
        """
        return keras.backend.squeeze(keras.backend.dot(x, keras.backend.expand_dims(kernel, axis=-1)), axis=-1)


class TemporalAttentionV2(keras.layers.Layer):

    def __init__(self,
                 w_regularizer=None,
                 b_regularizer=None,
                 w_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.w_constraint = keras.constraints.get(w_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.w = None
        self.b = None
        super(TemporalAttentionV2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        n_channels = input_shape[-1]
        self.w = self.add_weight(shape=(n_channels, n_channels),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(n_channels,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(TemporalAttentionV2, self).build(input_shape)

    def compute_mask(self, input_tensor, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # x: T*D , W: D*D ==> a: T*D
        a = self._dot_product(x, self.w)

        if self.bias:
            a += self.b

        a = keras.backend.tanh(a)

        alpha = keras.backend.exp(a)
        denominator = keras.backend.sum(alpha, axis=-1, keepdims=True) + keras.backend.epsilon()
        alpha /= denominator

        # x: T*D, alpha: T*1
        weighted_input = x * alpha
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def _dot_product(x, kernel):
        """Wrapper for dot product operation,

        Args:
            x: input
            kernel: weights
        """
        return keras.backend.dot(x, kernel)


class InstanceNormalization(keras.layers.Layer):
    """Instance normalization layer.

    Normalizes the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    of each feature map for each instance in batch close to 0 and the standard
    deviation close to 1.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv1D` layer with
            `data_format="channels_last"`,
            set `axis=-1` in `InstanceNorm`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        Input shape: Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a Sequential model.
        Output shape: Same shape as input.

    References:
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 mean=0,
                 stddev=1,
                 **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.mean = mean
        self.stddev = stddev
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = keras.layers.InputSpec(ndim=ndim)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean + self.mean) / stddev * self.stddev
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

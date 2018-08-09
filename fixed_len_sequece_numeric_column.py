"""Alternative implementation of tf.contrib.sequence_numeric_column for dense tensor."""

import collections

from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops


def fixed_len_sequence_numeric_column(key,
                                      shape=(1, ),
                                      default_value=0.,
                                      dtype=dtypes.float32,
                                      normalizer_fn=None):
    shape = fc._check_shape(shape=shape, key=key)
    if not (dtype.is_integer or dtype.is_floating):
        raise ValueError('dtype must be convertible to float. ' 'dtype: {}, key: {}'.format(dtype, key))
    if normalizer_fn is not None and not callable(normalizer_fn):
        raise TypeError('normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))

    return _FixedLenSequenceNumericColumn(
        key, shape=shape, default_value=default_value, dtype=dtype, normalizer_fn=normalizer_fn)


def _sequence_length_from_dense_tensor(dense_tensor):
    batch_size = array_ops.shape(dense_tensor)[:1]
    sequence_length = array_ops.shape(dense_tensor)[1:2]
    return array_ops.tile(sequence_length, batch_size)


class _FixedLenSequenceNumericColumn(
    fc._SequenceDenseColumn,
    collections.namedtuple('_SequenceFixedLenNumericColumn',
                           ['key', 'shape', 'default_value', 'dtype', 'normalizer_fn'])):
    """Represents sequences of numeric data."""

    @property
    def name(self):
        return self.key

    @property
    def _parse_example_spec(self):
        return {
            self.key: parsing_ops.FixedLenSequenceFeature(self.shape, self.dtype, allow_missing=True)
        }

    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        if self.normalizer_fn is not None:
            input_tensor = self.normalizer_fn(input_tensor)
        return input_tensor

    @property
    def _variable_shape(self):
        return tensor_shape.TensorShape(self.shape)

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        # Do nothing with weight_collections and trainable since no variables are
        # created in this function.
        del weight_collections
        del trainable

        dense_tensor = inputs.get(self)
        sequence_length = _sequence_length_from_dense_tensor(dense_tensor)

        return fc._SequenceDenseColumn.TensorSequenceLengthPair(
            dense_tensor=dense_tensor, sequence_length=sequence_length)
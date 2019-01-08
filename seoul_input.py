# TODO: We only need the last time step of labels, so the axis of time_length in labels can be removed.

import pandas as pd
import tensorflow as tf
import numpy as np
from functools import partial

from read_and_preprocess_pm import TargetPM
from fixed_len_sequece_numeric_column import fixed_len_sequence_numeric_column


def get_tf_sequence_feature_columns(features, normalizer_fns):
    """
    :param features: A dictionary of features {column_name: [batch_size, sequence_length]
    :param normalizer_fns: A dictionary of normalizer functions {column_name: normalizer_fn}
    :return: feature_columns: A list of tf.feature_column
    Keys of features and normalizer_fns must be same.
    """
    if not features.keys() == normalizer_fns.keys():
        raise Exception('Keys of features and normalizer_fns must be same.')

    # return [tf.contrib.feature_column.sequence_numeric_column(column_name, normalizer_fn=normalizer_fns[column_name])
    #         for column_name in features]
    return [fixed_len_sequence_numeric_column(column_name, normalizer_fn=normalizer_fns[column_name])
            for column_name in features]


def get_normalizer_fns(normalizer_fn, **normalizer_fn_params):
    all_param_column_names = [normalizer_fn_params[param].keys() for param in normalizer_fn_params]
    if not is_all_equal(all_param_column_names):
        raise Exception('Keys of normalizer_fn_params must be same.')

    normalizer_fns = {}
    column_names = all_param_column_names[0]
    for column_name in column_names:
        normalizer_fns[column_name] = partial(normalizer_fn, **{param: normalizer_fn_params[param][column_name]
                                                                for param in normalizer_fn_params})
    return normalizer_fns


def is_all_equal(iterable):
    return iterable.count(iterable[0]) == len(iterable)


def prepare_tf_dataset(df_features, df_labels):
    features = _cast_dataframe_to_dict(df_features)
    labels = _cast_dataframe_to_dict(df_labels)

    features_mean = df_features.mean()
    features_stddev = df_features.std()
    features_normalizer_fns = get_normalizer_fns(standardize, mean=features_mean, stddev=features_stddev)

    # label data for Seoul should be normalized by original feature columns
    labels_mean = {}
    labels_stddev = {}
    for column_name in df_labels:
        key, _ = TargetPM.get_key_hour_from_column_name(column_name)
        labels_mean[column_name] = features_mean[key]
        labels_stddev[column_name] = features_stddev[key]
    labels_normalizer_fns = get_normalizer_fns(standardize, mean=labels_mean, stddev=labels_stddev)

    feature_columns = get_tf_sequence_feature_columns(features, features_normalizer_fns)
    label_columns = get_tf_sequence_feature_columns(labels, labels_normalizer_fns)

    return features, labels, feature_columns, label_columns


def _cast_dataframe_to_dict(dataframe):
    casted_dict = dict(dataframe)
    for key in casted_dict:
        series = casted_dict[key]
        if isinstance(series.index, pd.MultiIndex):
            casted_dict[key] = _cast_multi_index_series_to_array(series)
    return casted_dict


def _cast_multi_index_series_to_array(multi_index_series, out_shape=None):
    if out_shape is None:
        out_shape = [len(multi_index_series.index.get_level_values(name).unique())
                     for name in multi_index_series.index.names]
    return multi_index_series.values.reshape(out_shape)


def standardize(dataset, mean, stddev):
    if np.isclose(stddev, 0):
        return dataset - mean
    return (dataset - mean) / stddev


def inverse_standardize(standardized_dataset, mean, stddev):
    if np.isclose(stddev, 0):
        return standardized_dataset + mean
    return standardized_dataset * stddev + mean


# TODO: Set dynamic and static shape of tensors after applying sliding window function
def sliding_window_input_fn(features, labels, window_size, batch_size, num_epochs):
    """
    :param features: dict of numpy array, [num_stations, time_length]
    :param labels: dict of numpy array, [num_stations, time_length]
    :param window_size: int
    :param batch_size: int
    :param num_epochs: int
    :return: tf.data.Dataset
    """
    dataset_features = tf.data.Dataset.from_tensor_slices(features)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)

    sliding_window_func = tf.contrib.data.sliding_window_batch(window_size=window_size)

    dataset_features = dataset_features.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x).apply(sliding_window_func))
    dataset_labels = dataset_labels.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x).apply(sliding_window_func))

    # Need to assert output_shapes because sliding_window_func forgets the TensorShape of its SlideDataset
    dataset_features = dataset_features.apply(
        tf.contrib.data.assert_element_shape({key: [window_size] for key in features}))
    dataset_labels = dataset_labels.apply(
        tf.contrib.data.assert_element_shape({key: [window_size] for key in labels}))

    dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))

    num_station, time_length = next(iter(features.values())).shape
    num_data_per_station = time_length - window_size + 1

    return dataset.shuffle(num_data_per_station).repeat(num_epochs).batch(batch_size)

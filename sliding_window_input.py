import tensorflow as tf


def my_input_fn(features, labels, window_size, batch_size):
    dataset_features = tf.data.Dataset.from_tensor_slices(features)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)

    sliding_window_func = tf.contrib.data.sliding_window_batch(window_size=window_size, stride=1)

    dataset_features = dataset_features.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x).apply(sliding_window_func))
    dataset_labels = dataset_labels.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x).apply(sliding_window_func))

    dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))

    return dataset.shuffle(1000).repeat(5).batch(batch_size)

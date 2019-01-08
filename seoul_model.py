# Here we feed data by tf.data and tf.feature_column APIs
# tf.feature_column.input_layer constructs tensors by fed data and feature columns
# Basically, labels do not need to be fed into input layer, but it's useful to construct label tensors(targets).
# TODO: Remove dependency general models and one for seoul
# TODO: Write an encoder model function to combine encoder part of simple_lstm and seq2seq models

import tensorflow as tf
import numpy as np
from fixed_len_sequece_numeric_column import fixed_len_sequence_numeric_column
from seoul_input import TargetPM
import seoul_transformer


def simple_cnn(features, labels, mode, params):
    features_mean, features_stddev = params['features_statistics']
    feature_columns = params['feature_columns']
    label_columns = params['label_columns']
    target_pm = params['target_pm']
    learning_rate = params['learning_rate']

    input_dim = len(feature_columns)
    output_dim = len(target_pm.label_columns)

    # label_columns should be sorted when you use tf.feature_column.input_layer for labels
    label_columns = sorted(label_columns, key=lambda x: x.name)

    # features and labels (dict): {label_column: [batch_size, sequence_length(input)]}
    # Shape of inputs and targets: [batch_size, sequence_length, feature_dim]
    inputs, _ = tf.contrib.feature_column.sequence_input_layer(features, feature_columns)
    targets, _ = tf.contrib.feature_column.sequence_input_layer(labels, label_columns)
    targets = targets[:, -1]  # Discard values except for the last time step
    
    inputs.set_shape([None, params['window_size'], input_dim])

    if params['conv_embedding']:
        with tf.variable_scope('conv_embedding'):
            large_inputs_embedder = SeoulLargeInputsEmbedder(
                params['day_region_start_hour'], params['day_region_num_layer'],
                params['week_region_start_hour'], params['week_region_num_layer'])
            inputs = large_inputs_embedder(inputs)

    input_shape = combined_static_and_dynamic_shape(inputs)
    inputs = tf.reshape(inputs, [input_shape[0], input_shape[1]*input_shape[2], 1])
    
    conv1 = tf.layers.conv1d(
        inputs=inputs,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=128,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=256,
        kernel_size=5,
        padding="same",
        activation=None)
    pool3_global_average = tf.reduce_mean(conv3, 1)

    # Dense Layer
    dense = tf.layers.dense(inputs=pool3_global_average, units=256, activation=tf.nn.relu)
    outputs = tf.layers.dense(inputs=dense, units=output_dim, activation=None)
    
    predictions = {}
    for i, column in enumerate(label_columns):
        key, _ = target_pm.get_key_hour_from_column_name(column.name)
        mean = features_mean[key]
        stddev = features_stddev[key]
        predictions[column.name] = _inverse_standardize(outputs[:, i], mean, stddev)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs)
    loss += add_l2_loss(tf.trainable_variables(), scale_factor=0.001)

    errors, eval_metric_ops = get_label_errors_and_metrics(labels, predictions, label_columns, target_pm)
    _add_summary_training_errors(errors)

    if mode == tf.estimator.ModeKeys.EVAL:
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[logging_hook])

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])    


def simple_dnn(features, labels, mode, params):
    features_mean, features_stddev = params['features_statistics']
    feature_columns = params['feature_columns']
    label_columns = params['label_columns']
    target_pm = params['target_pm']
    learning_rate = params['learning_rate']

    input_dim = len(feature_columns)
    output_dim = len(target_pm.label_columns)

    # label_columns should be sorted when you use tf.feature_column.input_layer for labels
    label_columns = sorted(label_columns, key=lambda x: x.name)

    # features and labels (dict): {label_column: [batch_size, sequence_length(input)]}
    # Shape of inputs and targets: [batch_size, sequence_length, feature_dim]
    inputs, _ = tf.contrib.feature_column.sequence_input_layer(features, feature_columns)
    targets, _ = tf.contrib.feature_column.sequence_input_layer(labels, label_columns)
    targets = targets[:, -1]  # Discard values except for the last time step

    inputs.set_shape([None, params['window_size'], input_dim])

    if params['conv_embedding']:
        with tf.variable_scope('conv_embedding'):
            large_inputs_embedder = SeoulLargeInputsEmbedder(
                params['day_region_start_hour'], params['day_region_num_layer'],
                params['week_region_start_hour'], params['week_region_num_layer'])
            inputs = large_inputs_embedder(inputs)

    input_shape = combined_static_and_dynamic_shape(inputs)
    inputs = tf.reshape(inputs, [input_shape[0], input_shape[1] * input_shape[2]])
    
    for units in [128, 64, 32]:
        inputs = tf.layers.dense(inputs, units, tf.nn.relu)
    outputs = tf.layers.dense(inputs, output_dim, activation=None)
    
    predictions = {}
    for i, column in enumerate(label_columns):
        key, _ = target_pm.get_key_hour_from_column_name(column.name)
        mean = features_mean[key]
        stddev = features_stddev[key]
        predictions[column.name] = _inverse_standardize(outputs[:, i], mean, stddev)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs)
    loss += add_l2_loss(tf.trainable_variables(), scale_factor=0.001)

    errors, eval_metric_ops = get_label_errors_and_metrics(labels, predictions, label_columns, target_pm)
    _add_summary_training_errors(errors)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[logging_hook])

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])    


def simple_lstm(features, labels, mode, params):
    features_mean, features_stddev = params['features_statistics']
    feature_columns = params['feature_columns']
    label_columns = params['label_columns']
    target_pm = params['target_pm']
    learning_rate = params['learning_rate']

    output_dim = len(target_pm.label_columns)

    # label_columns should be sorted when you use tf.feature_column.input_layer for labels
    label_columns = sorted(label_columns, key=lambda x: x.name)

    # features and labels (dict): {label_column: [batch_size, sequence_length(input)]}
    # Shape of inputs and targets: [batch_size, sequence_length, feature_dim]
    inputs, _ = tf.contrib.feature_column.sequence_input_layer(features, feature_columns)
    targets, _ = tf.contrib.feature_column.sequence_input_layer(labels, label_columns)
    targets = targets[:, -1]  # Discard values except for the last time step

    if params['conv_embedding']:
        with tf.variable_scope('conv_embedding'):
            large_inputs_embedder = SeoulLargeInputsEmbedder(
                params['day_region_start_hour'], params['day_region_num_layer'],
                params['week_region_start_hour'], params['week_region_num_layer'])
            inputs = large_inputs_embedder(inputs)

    with tf.variable_scope('pm_regression') as vs:
        # create stacked LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['num_encoder_states']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        outputs, state = tf.nn.dynamic_rnn(
            multi_rnn_cell, inputs, dtype=tf.float32)
        outputs = outputs[:, -1]  # Discard values except for the last time step
        outputs = tf.layers.dense(outputs, output_dim, kernel_initializer=tf.glorot_uniform_initializer())

        predictions = {}
        for i, column in enumerate(label_columns):
            key, _ = TargetPM.get_key_hour_from_column_name(column.name)
            mean = features_mean[key]
            stddev = features_stddev[key]
            predictions[column.name] = _inverse_standardize(outputs[:, i], mean, stddev)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.absolute_difference(labels=targets, predictions=outputs)
        loss += add_l2_loss(tf.trainable_variables(), scale_factor=0.0001)

        errors, eval_metric_ops = get_label_errors_and_metrics(labels, predictions, label_columns, target_pm)
        _add_summary_training_errors(errors)

        if mode == tf.estimator.ModeKeys.EVAL:
            logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                              evaluation_hooks=[logging_hook])

        # mode for tf.estimator.ModeKeys.TRAIN
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), decay_steps=1000,
                                                   decay_rate=0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])


def seq2seq(features, labels, mode, params):
    features_mean, features_stddev = params['features_statistics']
    feature_columns = params['feature_columns']
    target_pm = params['target_pm']
    learning_rate = params['learning_rate']

    output_dim = len(target_pm.keys)
    maximum_sequence_length = len(target_pm.hours)

    # features and labels (dict): {label_column: [batch_size, sequence_length(input)]}
    # labels should be transformed to label_sequences by combining columns of target_pm.hours
    # in addition to discard values except for the last time step
    label_sequences, label_sequences_columns = _transform_labels_to_sequences(labels, target_pm,
                                                                              features_mean, features_stddev)
    # label_sequences (dict): {target_pm.keys: [batch_size, sequence_length(output)]}

    # label_columns should be sorted when you use tf.feature_column.input_layer for labels
    label_sequences_columns = sorted(label_sequences_columns, key=lambda x: x.name)

    # Shape of inputs and targets: [batch_size, sequence_length, feature_dim]
    inputs, _ = tf.contrib.feature_column.sequence_input_layer(features, feature_columns)
    targets, targets_sequence_length = tf.contrib.feature_column.sequence_input_layer(label_sequences,
                                                                                      label_sequences_columns)

    if params['conv_embedding']:
        with tf.variable_scope('conv_embedding'):
            large_inputs_embedder = SeoulLargeInputsEmbedder(
                params['day_region_start_hour'], params['day_region_num_layer'],
                params['week_region_start_hour'], params['week_region_num_layer'])
            inputs = large_inputs_embedder(inputs)

    # batch_size can vary
    batch_size = tf.shape(inputs)[0]

    with tf.variable_scope('pm_regression') as vs:
        # Implementation of encoder
        # create stacked LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['num_encoder_states']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, dtype=tf.float32)
        # inputs : [batch, length, depth]

        # Implementation of decoder with attention wrapper
        attention_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(size)
                                                      for size in params['num_decoder_states']])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_cell.output_size, encoder_outputs)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism,
                                                           alignment_history=True)

        # start tokens like a <GO> symbol
        start_tokens, _ = tf.contrib.feature_column.sequence_input_layer(features, label_sequences_columns)
        start_tokens = start_tokens[:, -1]  # Discard values except for the last time step

        if mode == tf.estimator.ModeKeys.TRAIN:
            # axis=1 means the axis of sequence_length
            training_helper_inputs = tf.concat((tf.expand_dims(start_tokens, axis=1), targets[:, :-1]), axis=1)
            helper = tf.contrib.seq2seq.TrainingHelper(training_helper_inputs, targets_sequence_length)

        elif mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            end_fn_return = tf.constant(False, dtype=tf.bool)
            helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=tf.identity,
                                                        sample_shape=start_tokens.shape[1],
                                                        sample_dtype=tf.float32,
                                                        start_inputs=start_tokens,
                                                        end_fn=lambda x: end_fn_return)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell.zero_state(batch_size, tf.float32),
                                                  output_layer=tf.layers.Dense(
                                                      output_dim,
                                                      kernel_initializer=tf.glorot_uniform_initializer()))

        (decoder_outputs, decoder_state, _) = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_sequence_length)
        outputs = decoder_outputs.rnn_output  # Shape of [batch_size, sequence_length, feature_dim]
        attention_outputs = decoder_state.alignment_history.stack()  # Shape of [length, batch, depth]
        attention_outputs.set_shape(attention_outputs.get_shape().merge_with([maximum_sequence_length, None, None]))
        attention_mean = tf.reduce_mean(attention_outputs, 1)  # Shape of [length, depth]

        predictions = {}
        # predictions of key sequences
        for i, column in enumerate(label_sequences_columns):
            key = column.name
            mean = features_mean[key]
            stddev = features_stddev[key]
            predictions[key] = _inverse_standardize(outputs[:, :, i], mean, stddev)
        # predictions of all label columns
        for i, column in enumerate(label_sequences_columns):
            key = column.name
            for j, hour in enumerate(target_pm.hours):
                column_name = target_pm.get_label_column_name(key, hour)
                mean = features_mean[key]
                stddev = features_stddev[key]
                predictions[column_name] = _inverse_standardize(outputs[:, j, i], mean, stddev)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.absolute_difference(labels=targets, predictions=outputs)
        loss += add_l2_loss(tf.trainable_variables(), scale_factor=0.0001)

        errors, eval_metric_ops = get_all_errors_and_metrics(labels, label_sequences, predictions,
                                                             label_sequences_columns, target_pm)
        _add_summary_training_errors(errors)

        if mode == tf.estimator.ModeKeys.EVAL:
            logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                              evaluation_hooks=[logging_hook])

        # mode for tf.estimator.ModeKeys.TRAIN
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), decay_steps=1000,
                                                   decay_rate=0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])


def _transform_labels_to_sequences(labels, target_pm, features_mean, features_stddev):
    label_sequences = {}
    for key in target_pm.keys:
        label_sequences[key] = tf.concat([
            labels[column_name][:, -1:]  # Discard values except for the last time step
            for column_name in target_pm.get_label_column_names_by_key(key)
        ], axis=1)

    label_sequences_columns = [fixed_len_sequence_numeric_column(
        key, normalizer_fn=lambda x: _standardize(x, features_mean[key], features_stddev[key]))
        for key in target_pm.keys]

    return label_sequences, label_sequences_columns


def transformer(features, labels, mode, params):
    features_mean, features_stddev = params['features_statistics']
    feature_columns = params['feature_columns']
    target_pm = params['target_pm']
    learning_rate = params['learning_rate']

    # features and labels (dict): {label_column: [batch_size, sequence_length(input)]}
    # labels should be transformed to label_sequences by combining columns of target_pm.hours
    # in addition to discard values except for the last time step
    label_sequences, label_sequences_columns = _transform_labels_to_sequences(labels, target_pm,
                                                                              features_mean, features_stddev)
    # label_sequences (dict): {target_pm.keys: [batch_size, sequence_length(output)]}

    # label_columns should be sorted when you use tf.feature_column.input_layer for labels
    label_sequences_columns = sorted(label_sequences_columns, key=lambda x: x.name)

    # Shape of inputs and targets: [batch_size, sequence_length, feature_dim]
    inputs, _ = tf.contrib.feature_column.sequence_input_layer(features, feature_columns)
    targets, targets_sequence_length = tf.contrib.feature_column.sequence_input_layer(label_sequences,
                                                                                      label_sequences_columns)

    if params['conv_embedding']:
        with tf.variable_scope('conv_embedding'):
            large_inputs_embedder = SeoulLargeInputsEmbedder(
                params['day_region_start_hour'], params['day_region_num_layer'],
                params['week_region_start_hour'], params['week_region_num_layer'])
            inputs = large_inputs_embedder(inputs)

    params['input_size'] = inputs.shape[-1]
    params['output_size'] = targets.shape[-1]

    with tf.variable_scope("transformer"):
        model = seoul_transformer.SeoulTransformer(params, mode == tf.estimator.ModeKeys.TRAIN)
        outputs = model(inputs, targets)

    predictions = {}
    # predictions of key sequences
    for i, column in enumerate(label_sequences_columns):
        key = column.name
        mean = features_mean[key]
        stddev = features_stddev[key]
        predictions[key] = _inverse_standardize(outputs[:, :, i], mean, stddev)
    # predictions of all label columns
    for i, column in enumerate(label_sequences_columns):
        key = column.name
        for j, hour in enumerate(target_pm.hours):
            column_name = target_pm.get_label_column_name(key, hour)
            mean = features_mean[key]
            stddev = features_stddev[key]
            predictions[column_name] = _inverse_standardize(outputs[:, j, i], mean, stddev)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.absolute_difference(labels=targets, predictions=outputs)
    loss += add_l2_loss(tf.trainable_variables(), scale_factor=0.0001)

    errors, eval_metric_ops = get_all_errors_and_metrics(labels, label_sequences, predictions,
                                                         label_sequences_columns, target_pm)
    _add_summary_training_errors(errors)

    if mode == tf.estimator.ModeKeys.EVAL:
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[logging_hook])

    # mode for tf.estimator.ModeKeys.TRAIN
    learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), decay_steps=1000,
                                               decay_rate=0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # gradient clipping
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])


class LargeInputsEmbeddingLayer(tf.layers.Layer):
    def __init__(self, filter_size, kernel_size, num_layers):
        super(LargeInputsEmbeddingLayer, self).__init__()
        self._filter_size = filter_size
        self._kernel_size = kernel_size
        self._num_layers = num_layers
        self._convs = []
        self._pools = []

    def build(self, _):
        for i in range(self._num_layers):
            self._convs.append(tf.layers.Conv1D(filters=self._filter_size, kernel_size=self._kernel_size,
                                                padding='same', activation=tf.nn.relu))
            self._pools.append(tf.layers.MaxPooling1D(2, 2))

    def call(self, inputs):
        for i in range(self._num_layers):
            inputs = self._convs[i](inputs)
            inputs = self._pools[i](inputs)
        return inputs


class SeoulLargeInputsEmbedder(tf.layers.Layer):
    def __init__(self, day_region_start_hour, day_region_num_layer,
                 week_region_start_hour, week_region_num_layer):
        super(SeoulLargeInputsEmbedder, self).__init__()
        self._day_region_start_hour = day_region_start_hour
        self._day_region_num_layer = day_region_num_layer
        self._week_region_start_hour = week_region_start_hour
        self._week_region_num_layer = week_region_num_layer

    def build(self, _):
        self._day_embedding_layer = LargeInputsEmbeddingLayer(3, 10, self._day_region_num_layer)
        self._week_embedding_layer = LargeInputsEmbeddingLayer(3, 10, self._week_region_num_layer)

    def call(self, inputs):
        day_inputs = inputs[:, -self._day_region_start_hour:]
        week_inputs = inputs[:, -self._week_region_start_hour:-self._day_region_start_hour]
        inputs = tf.concat((self._week_embedding_layer(week_inputs), self._day_embedding_layer(day_inputs)), 1)
        return inputs


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
    tensor: A tensor of any type.
    Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def get_key_errors_and_metrics(label_sequences, predictions, label_sequences_columns):
    errors = {}
    eval_metric_ops = {}
    # errors of key sequences
    for i, column in enumerate(label_sequences_columns):
        key = column.name
        label_key = '{}/Avg'.format(key)
        errors[label_key] = _compute_mean_absolute_error(
            labels=label_sequences[key], predictions=predictions[key])
        eval_metric_ops[label_key] = tf.metrics.mean_absolute_error(
            labels=label_sequences[key], predictions=predictions[key])
    return errors, eval_metric_ops


def get_label_errors_and_metrics(labels, predictions, label_columns, target_pm):
    errors = {}
    eval_metric_ops = {}
    # errors of all label columns
    for i, column in enumerate(label_columns):
        key = column.name
        for j, hour in enumerate(target_pm.hours):
            column_name = target_pm.get_label_column_name(key, hour)
            label_key = '{}/{}h'.format(key, hour)
            errors[label_key] = _compute_mean_absolute_error(
                labels=labels[column_name][:, -1], predictions=predictions[column_name])
            eval_metric_ops[label_key] = tf.metrics.mean_absolute_error(
                labels=labels[column_name][:, -1], predictions=predictions[column_name])
    return errors, eval_metric_ops


# TODO: Remove target_pm dependency
def get_all_errors_and_metrics(labels, label_sequences, predictions, label_sequences_columns, target_pm):
    errors_key, eval_metric_ops_key = get_key_errors_and_metrics(
        label_sequences, predictions, label_sequences_columns)
    errors_label, eval_metric_ops_label = get_label_errors_and_metrics(
        labels, predictions, label_sequences_columns, target_pm)
    errors = {**errors_key, **errors_label}
    eval_metric_ops = {**eval_metric_ops_key, **eval_metric_ops_label}
    return errors, eval_metric_ops


def _add_summary_training_errors(errors):
    for label_key in errors:
        tf.summary.scalar(label_key, errors[label_key])


def _standardize(dataset, mean, stddev):
    if np.isclose(stddev, 0):
        return dataset - mean
    return (dataset - mean) / stddev


def _inverse_standardize(standardized_dataset, mean, stddev):
    if np.isclose(stddev, 0):
        return standardized_dataset + mean
    return standardized_dataset * stddev + mean


def _compute_mean_absolute_error(labels, predictions):
    return tf.reduce_mean(tf.abs(labels - predictions))


def add_l2_loss(variables, scale_factor):
    l2_loss = 0
    for v in variables:
        if 'bias' not in v.name.lower():
            l2_loss += scale_factor * tf.nn.l2_loss(v)
    return l2_loss
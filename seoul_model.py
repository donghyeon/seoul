# Here we feed data by tf.data and tf.feature_column APIs
# tf.feature_column.input_layer constructs tensors by fed data and feature columns
# Basically, labels do not need to be fed into input layer, but it's useful to construct label tensors(targets).
# TODO: Remove dependency general models and one for seoul
# TODO: Write an encoder model function to combine encoder part of simple_lstm and seq2seq models

import tensorflow as tf


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

    with tf.variable_scope('pm_regression', reuse=tf.AUTO_REUSE) as vs:
        # create stacked LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['num_encoder_states']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        outputs, state = tf.nn.dynamic_rnn(
            multi_rnn_cell, inputs, dtype=tf.float32)
        outputs = outputs[:, -1]  # Discard values except for the last time step

        outputs = tf.layers.dense(outputs, output_dim)
        predictions = {}
        for i, column in enumerate(label_columns):
            key, _ = target_pm.get_key_hour_from_column_name(column.name)
            mean = features_mean[key]
            stddev = features_stddev[key]
            predictions[column.name] = _inverse_standardize(outputs[..., i], mean, stddev)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.absolute_difference(labels=targets, predictions=outputs)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        # mode for tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


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

    # batch_size can vary
    batch_size = tf.shape(inputs)[0]

    with tf.variable_scope('pm_regression', reuse=tf.AUTO_REUSE) as vs:
        # Implementation of encoder
        # create stacked LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['num_encoder_states']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, dtype=tf.float32)
        # inputs : [batch, length, depth]

        # Implementation of decoder with attention wrapper
        attention_cell = tf.nn.rnn_cell.BasicLSTMCell(params['num_decoder_states'])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(params['num_decoder_states'], encoder_outputs)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism)

        # start tokens like a <GO> symbol
        start_tokens, _ = tf.contrib.feature_column.sequence_input_layer(features, label_sequences_columns)
        start_tokens = start_tokens[:, -1]  # Discard values except for the last time step

        if mode == tf.estimator.ModeKeys.TRAIN:
            # axis=1 means the axis of sequence_length
            training_helper_inputs = tf.concat((tf.expand_dims(start_tokens, axis=1), targets), axis=1)
            helper = tf.contrib.seq2seq.TrainingHelper(training_helper_inputs, targets_sequence_length)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=tf.identity,
                                                        sample_shape=tf.TensorShape([batch_size]),
                                                        sample_dtype=tf.float32,
                                                        start_inputs=start_tokens,
                                                        end_fn=tf.no_op)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell.zero_state(batch_size, tf.float32),
                                                  output_layer=tf.layers.Dense(output_dim))

        (decoder_outputs, _, _) = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                    maximum_iterations=maximum_sequence_length)
        outputs = decoder_outputs.rnn_output  # Shape of [batch_size, sequence_length, feature_dim]

        predictions = {}
        # predictions of key sequences
        for i, column in enumerate(label_sequences_columns):
            key = column.name
            mean = features_mean[key]
            stddev = features_stddev[key]
            predictions[key] = _inverse_standardize(outputs[..., i], mean, stddev)
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

        errors = {}
        # errors of key sequences
        for i, column in enumerate(label_sequences_columns):
            key = column.name
            errors[key] = _compute_mean_absolute_error(labels=label_sequences[key], predictions=predictions[key])
        # errors of all label columns
        for i, column in enumerate(label_sequences_columns):
            key = column.name
            for j, hour in enumerate(target_pm.hours):
                column_name = target_pm.get_label_column_name(key, hour)
                errors[column_name] = _compute_mean_absolute_error(
                    labels=labels[column_name][:, -1], predictions=predictions[column_name])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        # mode for tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, **errors}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def _transform_labels_to_sequences(labels, target_pm, features_mean, features_stddev):
    label_sequences = {}
    for key in target_pm.keys:
        label_sequences[key] = tf.concat([
            labels[column_name][:, -1:]  # Discard values except for the last time step
            for column_name in target_pm.get_label_column_names_by_key(key)
        ], axis=1)

    label_sequences_columns = [tf.contrib.feature_column.sequence_numeric_column(
        key, normalizer_fn=lambda x: _standardize(x, features_mean[key], features_stddev[key]))
        for key in target_pm.keys]

    return label_sequences, label_sequences_columns


def _standardize(dataset, mean, stddev):
    return (dataset - mean) / stddev


def _inverse_standardize(standardized_dataset, mean, stddev):
    return standardized_dataset * stddev + mean


def _compute_mean_absolute_error(labels, predictions):
    return tf.reduce_mean(tf.abs(labels - predictions))

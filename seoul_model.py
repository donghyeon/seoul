import tensorflow as tf


def simple_lstm(features, labels, mode, params):
    inputs = features
    targets = labels[:, -1, :]  # Discard values except for the last time step
    target_dict = params['target_dict']
    learning_rate = params['learning_rate']

    out_size = len([_ for target_column in target_dict
                    for _ in target_dict[target_column]])

    with tf.variable_scope('pm_regression', reuse=tf.AUTO_REUSE) as vs:
        # create stacked LSTMCells
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params['num_LSTM_states']]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        outputs, state = tf.nn.dynamic_rnn(
            multi_rnn_cell, inputs, dtype=tf.float32)  # inputs : [batch, length, depth]
        outputs = outputs[:, -1, :]  # Discard values except for the last time step

        pm_prediction = tf.layers.dense(outputs, out_size)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {}
            target_keys = ['%s_%dh' % (target_column, hour)
                           for target_column in target_dict
                           for hour in target_dict[target_column]]

            for i, target_key in enumerate(target_keys):
                predictions[target_key] = pm_prediction[:, i]
            predictions['pm_prediction'] = pm_prediction
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.absolute_difference(labels=targets, predictions=pm_prediction)

        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=100)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def seq2seq(features, labels, mode, params):
    # TODO: Change the shape of all tensors by [batch, length, depth] consistently.
    inputs = features
    targets = tf.expand_dims(labels[:, -1, :], -1)
    target_dict = params['target_dict']
    learning_rate = params['learning_rate']

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

        output_size = len(target_dict)
        output_sequence_length = params['output_sequence_length']

        batch_size = tf.shape(inputs)[0]
        start_tokens = tf.expand_dims(tf.expand_dims(
            features[:, -1, params['pm_index']], -1), -1)  # start tokens like a <GO> symbol
        if mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.TrainingHelper(tf.concat((start_tokens, targets), axis=1),
                                                       tf.fill([batch_size], output_sequence_length))
        elif mode == tf.estimator.ModeKeys.PREDICT:
            helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=tf.identity,
                                                        sample_shape=tf.TensorShape([batch_size]),
                                                        sample_dtype=tf.float32,
                                                        start_inputs=start_tokens,
                                                        end_fn=tf.no_op)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell.zero_state(batch_size, tf.float32),
                                                  output_layer=tf.layers.Dense(output_size))

        (decoder_outputs, _, _) = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=output_sequence_length)
        predictions = decoder_outputs.rnn_output

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.absolute_difference(targets, predictions)
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=100)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
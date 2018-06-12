import tensorflow as tf


def my_model_fn(features, labels, mode, params):
    inputs = features
    targets = labels[:, -1, :]  # Discard values except for the last time step
    target_dict = params['target_dict']
    learning_rate = params['learning_rate']

    out_size = len([_ for target_column in target_dict
                    for _ in target_dict[target_column]])

    with tf.variable_scope('pm_regression', reuse=tf.AUTO_REUSE) as vs:
        # create 2 LSTMCells
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

        loss = tf.losses.mean_squared_error(labels=targets, predictions=pm_prediction)

        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=100)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

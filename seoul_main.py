import time
import pandas as pd

from absl import flags
import tensorflow as tf

import read_and_preprocess_pm
from read_and_preprocess_pm import TargetPM
import seoul_input
import seoul_model

# Estimator flags
flags.DEFINE_string('model_dir', None, 'Path to output model directory.')
flags.DEFINE_integer('save_checkpoints_steps', 100, 'Steps to save a checkpoint.')
flags.DEFINE_integer('num_epochs', 10, 'Number of train epochs.')
# flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')  # Currently not supported
flags.DEFINE_integer('start_delay_secs', 5, 'Seconds not to evaluate after running this script.')
flags.DEFINE_integer('throttle_secs', 5, 'Seconds not to evaluate after the previous evaluation.')
flags.DEFINE_bool('evaluate_once', False, 'Whether to evaluate saved model once.')

# Optimizer flags
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate of an optimizer.')
flags.DEFINE_integer('batch_size', 128, 'Number of examples in a batch')

# Model flags
flags.DEFINE_string('model', None, 'Model to train (dnn, cnn, rnn, seq2seq, transformer).')
flags.DEFINE_string('target_keys', 'PM10,PM25', 'Labels to predict. Use comma for multiple keys.')
flags.DEFINE_string('target_hours', '3,6,12,24', 'Hours to predict. Use comma for multiple hours.')
flags.DEFINE_integer('window_size', 24, 'Window size of a sliding window input function.')
flags.DEFINE_bool('input_embedding', False, 'Whether to apply a 2-region convolutional input embedding.')
# flags.DEFINE_string(  # Currently not supported
#     'hparams_overrides', None,
#     'Hyperparameter overrides, represented as a string containing comma-separated hparam_name=value pairs.')
FLAGS = flags.FLAGS


def parse_string_by_commas(string):
    return string.split(',')


def main(unused_argv):
    # Load raw data
    # df_pm = read_and_preprocess_pm.read_pm_dataset('data/airkorea')

    # Or instead save and load pickled data
    # df_pm.to_pickle('df_pm.pickle')
    df_pm = pd.read_pickle('df_pm.pickle')
    df_pm = read_and_preprocess_pm.get_dataframe_with_complete_pm25(df_pm)

    # Fill proper values (by forward fill and backward fill methods) to replace missing values
    df_pm = read_and_preprocess_pm.treat_nan_by_fill_methods(df_pm)

    # Make output values to predict
    target_keys = parse_string_by_commas(FLAGS.target_keys)
    target_hours = list(map(int, parse_string_by_commas(FLAGS.target_hours)))
    target_pm = TargetPM(target_keys, target_hours)
    df_pm = read_and_preprocess_pm.make_target_values(df_pm, target_pm)

    # Preprocess dataframes
    df_pm, df_features, df_labels = read_and_preprocess_pm.preprocess_pm(df_pm, target_pm)

    dates_to_drop = df_pm.index.get_level_values('측정일시').unique()[0:0]
    # dates_to_drop = dates_to_drop[dates_to_drop % 3 != 1]
    stations_to_drop = df_pm.index.get_level_values('측정소코드').unique()[0:0]
    columns_to_drop = df_features.columns.drop([TargetPM.PM10, TargetPM.PM25])

    df_pm = df_pm.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
    df_features = df_features.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
    df_labels = df_labels.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드')


    df_features_train, df_labels_train, df_features_eval, df_labels_eval = \
        read_and_preprocess_pm.split_data_to_train_eval(df_features, df_labels)

    # TODO: Make an input function for tensorflow dataset
    # Prepare tensorflow dataset
    features_train, labels_train, feature_columns, label_columns = seoul_input.prepare_tf_dataset(df_features_train, df_labels_train)
    features_eval, labels_eval, _, _ = seoul_input.prepare_tf_dataset(df_features_eval, df_labels_eval)

    # hyper-parameters for encoder
    num_encoder_states = [64]

    # hyper-parameters for seq2seq
    num_decoder_states = [64]

    # TODO: Add FLAGS validator
    # Set model_dir as current time string if it was not set.
    if FLAGS.model_dir is None:
        current_time_string = time.strftime('%y%m%d_%H%M%S', time.localtime())
        print('FLAGS.model_dir was not set. Current time will be used as a model directory.')
        FLAGS.model_dir = current_time_string
    # TODO: Add an exception when user want to evaluate a pretrained model from FLAGS.model_dir
    # Set model as rnn if it was not set.
    if FLAGS.model is None:
        print('FLAGS.model was not set. Simple RNN is used for this training.')
        FLAGS.model = 'rnn'

    # Define an estimator object
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                        session_config=session_config,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    params = {'target_pm': target_pm, 'feature_columns': feature_columns, 'label_columns': label_columns,
              'features_statistics': read_and_preprocess_pm.get_statistics_for_standardization(df_features_train),
              'batch_size': FLAGS.batch_size, 'window_size': FLAGS.window_size,
              'num_encoder_states': num_encoder_states, 'num_decoder_states': num_decoder_states,
              'learning_rate': FLAGS.learning_rate,
              'conv_embedding': FLAGS.input_embedding,
              'day_region_start_hour': 24, 'day_region_num_layer': 1,
              'week_region_start_hour': 24 * 9, 'week_region_num_layer': 4,
              'sequence_length': len(target_pm.hours)}

    model_fn_dict = {'dnn': seoul_model.simple_dnn,
                     'cnn': seoul_model.simple_cnn,
                     'rnn': seoul_model.simple_lstm,
                     'seq2seq': seoul_model.seq2seq,
                     'transformer': seoul_model.transformer}

    seoul_regressor = tf.estimator.Estimator(
        model_fn=model_fn_dict[FLAGS.model], config=run_config,
        params={**params,
                'initializer_gain': 1.0, 'hidden_size': 64, 'layer_postprocess_dropout': 0.1,
                'num_heads': 8, 'attention_dropout': 0.1, 'relu_dropout': 0.1, 'allow_ffn_pad': True,
                'num_hidden_layers': 6, 'filter_size': 64})

    # Let's train and evaluate
    def train_input_fn():
        return seoul_input.sliding_window_input_fn(
            features_train, labels_train, FLAGS.window_size, FLAGS.batch_size, FLAGS.num_epochs)

    def eval_input_fn():
        return seoul_input.sliding_window_input_fn(
            features_eval, labels_eval, FLAGS.window_size, FLAGS.batch_size, 1)

    if FLAGS.evaluate_once:
        seoul_regressor.evaluate(eval_input_fn)
    else:
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          start_delay_secs=FLAGS.start_delay_secs,
                                          throttle_secs=FLAGS.throttle_secs)
        tf.estimator.train_and_evaluate(seoul_regressor, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()

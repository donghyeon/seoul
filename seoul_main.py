import time
import os
import json

import pandas as pd
import tensorflow as tf

import read_and_preprocess_pm
from read_and_preprocess_pm import TargetPM
import seoul_input
import seoul_model


# Load raw data
# df_pm = read_and_preprocess_pm.read_pm_dataset('/home/donghyeon/disk1/dataset/seoul/airkorea')

# Or instead save and load pickled data
# df_pm.to_pickle('df_pm.pickle')
df_pm = pd.read_pickle('df_pm.pickle')
df_pm = read_and_preprocess_pm.get_dataframe_with_complete_pm25(df_pm)

# Fill proper values (by forward fill and backward fill methods) to replace missing values
df_pm = read_and_preprocess_pm.treat_nan_by_fill_methods(df_pm)

# Make output values to predict
target_pm = TargetPM([TargetPM.PM10, TargetPM.PM25], [1, 2, 4, 8, 16, 24, 48, 72])
df_pm = read_and_preprocess_pm.make_target_values(df_pm, target_pm)

# Preprocess dataframes
df_pm, df_features, df_labels = read_and_preprocess_pm.preprocess_pm(df_pm, target_pm)

dates_to_drop = df_pm.index.get_level_values('측정일시').unique()[0:0]
# dates_to_drop = dates_to_drop[dates_to_drop % 3 != 1]
stations_to_drop = df_pm.index.get_level_values('측정소코드').unique()[0:0]
columns_to_drop = df_features.columns.drop([TargetPM.PM10, TargetPM.PM25])[0:0]

df_pm = df_pm.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
df_features = df_features.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
df_labels = df_labels.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드')


df_features_train, df_features_eval, df_features_test, df_labels_train, df_labels_eval, df_labels_test = \
    read_and_preprocess_pm.split_data_to_train_eval_test(df_features, df_labels)

# Prepare tensorflow dataset
features_train, labels_train, feature_columns, label_columns = seoul_input.prepare_tf_dataset(df_features_train, df_labels_train)
features_eval, labels_eval, _, _ = seoul_input.prepare_tf_dataset(df_features_eval, df_labels_eval)
features_test, labels_test, _, _ = seoul_input.prepare_tf_dataset(df_features_test, df_labels_test)


# TODO: Add an argument parser
# Set hyper-parameters for encoder
num_encoder_states = [128]
window_size = 24 * 9
batch_size = 10

# hyper-parameters for seq2seq
num_decoder_states = [128]

# hyper-parameters for training
learning_rate = 1e-4
num_epoch = 1


# Use specific directories to manage experiments
experiment_dir = 'experiments'
exp_time_str = time.strftime('%y%m%d_%H%M%S', time.localtime())
model_dir = os.path.join(experiment_dir, exp_time_str)


# Define an estimator object
tf.logging.set_verbosity(tf.logging.INFO)
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=session_config,
                                    save_checkpoints_steps=100)
params = {'target_pm': target_pm, 'feature_columns': feature_columns, 'label_columns': label_columns,
          'features_statistics': read_and_preprocess_pm.get_statistics_for_standardization(df_features_train),
          'batch_size': batch_size, 'window_size': window_size,
          'num_encoder_states': num_encoder_states, 'num_decoder_states': num_decoder_states,
          'learning_rate': learning_rate,
          'conv_embedding': True,
          'day_region_start_hour': 24, 'day_region_num_layer': 1,
          'week_region_start_hour': 24 * 9, 'week_region_num_layer': 4}
# seoul_regressor = tf.estimator.Estimator(
#     model_fn=seoul_model.simple_lstm, config=run_config,
#     params=params)

seoul_regressor = tf.estimator.Estimator(
    model_fn=seoul_model.seq2seq, config=run_config,
    params=params)
#
# seoul_regressor = tf.estimator.Estimator(
#     model_fn=seoul_model.transformer, config=run_config,
#     params={**params,
#             'initializer_gain': 1.0, 'hidden_size': 64, 'layer_postprocess_dropout': 0.1,
#             'num_heads': 8, 'attention_dropout': 0.1, 'relu_dropout': 0.1, 'allow_ffn_pad': True,
#             'num_hidden_layers': 6, 'filter_size': 64})


# Let's train and evaluate
train_spec = tf.estimator.TrainSpec(input_fn=lambda: seoul_input.sliding_window_input_fn(
    features_train, labels_train, window_size, batch_size, num_epoch))
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: seoul_input.sliding_window_input_fn(
    features_eval, labels_eval, window_size, batch_size, 1),
    start_delay_secs=60, throttle_secs=60)
tf.estimator.train_and_evaluate(seoul_regressor, train_spec, eval_spec)

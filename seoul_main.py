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
target_pm = TargetPM([TargetPM.PM10, TargetPM.PM25], [3, 6, 9, 12])
df_pm = read_and_preprocess_pm.make_target_values(df_pm, target_pm)

# Preprocess dataframes
df_pm, df_features, df_labels = read_and_preprocess_pm.preprocess_pm(df_pm, target_pm)

df_features_train, df_features_eval, df_features_test, df_labels_train, df_labels_eval, df_labels_test = \
    read_and_preprocess_pm.split_data_to_train_eval_test(df_features, df_labels)

# dates_to_drop = df_pm.index.get_level_values('측정일시').unique()[0:0]
# #dates_to_drop = dates_to_drop[dates_to_drop % 3 != 1]
# stations_to_drop = df_pm.index.get_level_values('측정소코드').unique()[0:0]
# columns_to_drop = df_features.columns.drop([TargetPM.PM10, TargetPM.PM25])[0:0]
#
# df_pm = df_pm.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
# df_features = df_features.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드').drop(columns_to_drop, axis=1)
# df_labels = df_labels.drop(dates_to_drop, level='측정일시').drop(stations_to_drop, level='측정소코드')

# Prepare tensorflow dataset
features_train, labels_train, feature_columns, label_columns = seoul_input.prepare_tf_dataset(df_features_train, df_labels_train)
features_eval, labels_eval, _, _ = seoul_input.prepare_tf_dataset(df_features_eval, df_labels_eval)
features_test, labels_test, _, _ = seoul_input.prepare_tf_dataset(df_features_test, df_labels_test)


# TODO: Add an argument parser
# Set hyper-parameters for encoder
num_encoder_states = [128]
window_size = 24
batch_size = 100

# hyper-parameters for seq2seq
num_decoder_states = [128]

# hyper-parameters for training
learning_rate = 0.01
num_epoch = 100

tf.logging.set_verbosity(tf.logging.INFO)

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

# Use specific directories to manage experiments
model_dir = 'models'
exp_prefix = 'seoul_lstm'
exp_time_str = time.strftime('%y%m%d_%H%M%S', time.localtime())
ckpt_dir = os.path.join(model_dir, exp_prefix, exp_time_str)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(os.path.join(model_dir, exp_prefix)):
    os.mkdir(os.path.join(model_dir, exp_prefix))


# Define an estimator object
# seoul_regressor = tf.estimator.Estimator(
#     model_fn=seoul_model.simple_lstm,
#     config=tf.estimator.RunConfig(model_dir=ckpt_dir, session_config=session_config),
#     params={'target_pm': target_pm, 'feature_columns': feature_columns, 'label_columns': label_columns,
#             'features_statistics': read_and_preprocess_pm.get_statistics_for_standardization(df_features),
#             'batch_size': batch_size, 'window_size': window_size,
#             'num_encoder_states': num_encoder_states,
#             'learning_rate': learning_rate})

seoul_regressor = tf.estimator.Estimator(
    model_fn=seoul_model.seq2seq,
    config=tf.estimator.RunConfig(model_dir=ckpt_dir, session_config=session_config),
    params={'target_pm': target_pm, 'feature_columns': feature_columns, 'label_columns': label_columns,
            'features_statistics': read_and_preprocess_pm.get_statistics_for_standardization(df_features),
            'batch_size': batch_size, 'window_size': window_size,
            'num_encoder_states': num_encoder_states, 'num_decoder_states': num_decoder_states,
            'learning_rate': learning_rate})

# seoul_regressor = tf.estimator.Estimator(
#     model_fn=seoul_model.transformer,
#     config=tf.estimator.RunConfig(model_dir=ckpt_dir, session_config=session_config),
#     params={'target_pm': target_pm, 'feature_columns': feature_columns, 'label_columns': label_columns,
#             'features_statistics': read_and_preprocess_pm.get_statistics_for_standardization(df_features),
#             'batch_size': batch_size, 'window_size': window_size,
#             'num_encoder_states': num_encoder_states, 'num_decoder_states': num_decoder_states,
#             'learning_rate': learning_rate,
#             'initializer_gain': 1.0, 'hidden_size': 64, 'layer_postprocess_dropout': 0.1,
#             'num_heads': 8, 'attention_dropout': 0.1, 'relu_dropout': 0.1, 'allow_ffn_pad': True,
#             'num_hidden_layers': 6, 'filter_size': 64})


# TODO: Find a popular library for adjusting configs of experiments or make this as a simple tool
# Load and save experiments' configs to manage experiments later
exp_configs = {}
exp_configs_filename = os.path.join(model_dir, exp_prefix + '.json')
if os.path.exists(exp_configs_filename):
    with open(exp_configs_filename, 'r') as f:
        exp_configs = json.load(f)
exp_configs[exp_time_str] = {'target_pm': target_pm.to_dict(), 'num_LSTM_states': num_encoder_states,
                             'learning_rate': learning_rate, 'batch_size': batch_size, 'window_size': window_size}
with open(exp_configs_filename, 'w') as f:
    json.dump(exp_configs, f)

# Let's train!
# seoul_regressor.train(input_fn=lambda: seoul_input.sliding_window_input_fn(
#     features_train, labels_train, window_size, batch_size, num_epoch))

train_spec = tf.estimator.TrainSpec(input_fn=lambda: seoul_input.sliding_window_input_fn(
    features_train, labels_train, window_size, batch_size, num_epoch))
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: seoul_input.sliding_window_input_fn(
    features_eval, labels_eval, window_size, batch_size, 1))

tf.estimator.train_and_evaluate(seoul_regressor, train_spec, eval_spec)

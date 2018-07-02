import time
import os
import json

import pandas as pd
import numpy as np
import tensorflow as tf

import read_and_preprocess_pm
import seoul_model
import sliding_window_input

# Load raw data
df_pm = read_and_preprocess_pm.read_pm_dataset('/home/donghyeon/disk1/dataset/seoul/airkorea')

# Or instead save and load pickled data
# df_pm.to_pickle('df_pm.pickle')
# df_pm = pd.read_pickle('df_pm.pickle')

# Make output values to predict
target_dict = {'PM10': [3, 6, 12, 24]}  # TODO: Split dict target_keys and target_hours for seq2seq learning
df_pm = read_and_preprocess_pm.make_target_values(df_pm, target_dict)

# Get tensors for training
pm_features, pm_labels, df_pm = read_and_preprocess_pm.preprocess_pm(df_pm, target_dict)
pm_features = read_and_preprocess_pm.treat_nan_by_interpolation(pm_features, df_pm).astype(np.float32)
pm_labels = read_and_preprocess_pm.treat_nan_by_interpolation(pm_labels, df_pm).astype(np.float32)


# TODO: Split train/val/test data


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


# TODO: Add an argument parser
# Set hyper-parameters for encoder
num_LSTM_states = [64, 64]
window_size = 24
batch_size = 64
learning_rate = 0.01

# hyper-parameters for seq2seq
target_key = list(target_dict.keys())[0]
output_sequence_length = len(target_dict[target_key])
pm_index = df_pm.columns.get_loc(target_key)
num_decoder_states = 64

# Define an estimator object
# seoul_regressor = tf.estimator.Estimator(
#     model_fn=seoul_model.simple_lstm,
#     config=tf.estimator.RunConfig(model_dir=ckpt_dir, session_config=session_config),
#     params={'target_dict': target_dict, 'num_LSTM_states': num_LSTM_states, 'learning_rate': learning_rate})

seoul_regressor = tf.estimator.Estimator(
    model_fn=seoul_model.seq2seq,
    config=tf.estimator.RunConfig(model_dir=ckpt_dir, session_config=session_config),
    params={'target_dict': target_dict, 'num_encoder_states': num_LSTM_states, 'learning_rate': learning_rate,
            'num_decoder_states': num_decoder_states, 'output_sequence_length': output_sequence_length,
            'pm_index': pm_index})

# TODO: Find a popular library for adjusting configs of experiments or make this as a simple tool
# Load and save experiments' configs to manage experiments later
exp_configs = {}
exp_configs_filename = os.path.join(model_dir, exp_prefix + '.json')
if os.path.exists(exp_configs_filename):
    with open(exp_configs_filename, 'r') as f:
        exp_configs = json.load(f)
exp_configs[exp_time_str] = {'target_dict': target_dict, 'num_LSTM_states': num_LSTM_states,
                             'learning_rate': learning_rate, 'batch_size': batch_size, 'window_size': window_size}
with open(exp_configs_filename, 'w') as f:
    json.dump(exp_configs, f)

# Let's train!
seoul_regressor.train(input_fn=lambda: sliding_window_input.my_input_fn(
    pm_features, pm_labels, window_size, batch_size))

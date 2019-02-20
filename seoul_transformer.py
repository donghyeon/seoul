import tensorflow as tf
from official.transformer.model.transformer import EncoderStack
from official.transformer.model.transformer import DecoderStack
from official.transformer.model import model_utils

_NEG_INF = -1e9


class SeoulTransformer(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params

        self.encoder_embedding_layer = tf.keras.layers.Dense(params['hidden_size'], name='encoder_embedding_layer')
        self.decoder_embedding_layer = tf.keras.layers.Dense(params['hidden_size'], name='decoder_embedding_layer')
        self.output_embedding_layer = tf.keras.layers.Dense(params['output_size'], name='output_embedding_layer')

        self.encoder_stack = EncoderStack(params, train)
        self.decoder_stack = DecoderStack(params, train)

    def __call__(self, inputs, start_tokens, targets=None):
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            attention_bias = seoul_get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias)

            if targets is None:
                return self.predict(start_tokens, encoder_outputs, attention_bias)
            else:
                outputs = self.decode(start_tokens, targets, encoder_outputs, attention_bias)
                return outputs

    def encode(self, inputs, attention_bias):
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.encoder_embedding_layer(inputs)
            inputs_padding = seoul_get_padding(inputs)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params["layer_postprocess_dropout"])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, start_tokens, targets, encoder_outputs, attention_bias):
        with tf.name_scope("decode"):
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.concat([start_tokens, targets[:, :-1]], axis=1)
                decoder_inputs = self.decoder_embedding_layer(decoder_inputs)
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
            if self.train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params["layer_postprocess_dropout"])

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length)
            decoder_outputs = self.decoder_stack(
                decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias)
            outputs = self.output_embedding_layer(decoder_outputs)
            return outputs

    def predict(self, start_tokens, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        with tf.name_scope('decode'):
            batch_size = tf.shape(encoder_outputs)[0]
            max_decode_length = self.params['sequence_length']
            timing_signal = model_utils.get_position_encoding(max_decode_length, self.params['hidden_size'])
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)

            # Create cache storing decoder attention values for each layer.
            cache = {
                'layer_%d' % layer: {
                    'k': tf.zeros([batch_size, 0, self.params['hidden_size']]),
                    'v': tf.zeros([batch_size, 0, self.params['hidden_size']])
                } for layer in range(self.params['num_hidden_layers'])}

            # Add encoder output and attention bias to the cache.
            cache['encoder_outputs'] = encoder_outputs
            cache['encoder_decoder_attention_bias'] = encoder_decoder_attention_bias

            # Forward decoder_inputs to decoder_stack max_decode_length times instead of applying beam search.
            decoder_outputs = tf.zeros([batch_size, 0, self.params['output_size']])
            decoder_inputs = start_tokens
            for i in range(max_decode_length):
                decoder_inputs = self.decoder_embedding_layer(decoder_inputs)
                decoder_inputs += timing_signal[i:i + 1]
                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
                decoder_inputs = self.decoder_stack(
                    decoder_inputs, cache.get('encoder_outputs'), self_attention_bias,
                    cache.get('encoder_decoder_attention_bias'), cache)
                decoder_inputs = self.output_embedding_layer(decoder_inputs)
                decoder_outputs = tf.concat([decoder_outputs, decoder_inputs], axis=1)
        return decoder_outputs


def seoul_get_padding(x):
    with tf.name_scope("padding"):
        return tf.zeros(shape=tf.shape(x)[:-1])


def seoul_get_padding_bias(x):
    with tf.name_scope("attention_bias"):
        padding = seoul_get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias

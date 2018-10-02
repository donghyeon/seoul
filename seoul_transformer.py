import tensorflow as tf
from official.transformer.model.transformer import EncoderStack
from official.transformer.model.transformer import DecoderStack
from official.transformer.model import model_utils

_NEG_INF = -1e9

class SeoulEmbeddingLayer(tf.layers.Layer):
    def __init__(self, hidden_size):
        super(SeoulEmbeddingLayer, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.embedding_layer = tf.layers.Dense(self.hidden_size, name='embedding_layer')

        self.built = True

    def call(self, x):
        with tf.name_scope("embedding"):
            return self.embedding_layer(x)


class SeoulTransformer(object):
    def __init__(self, params, train):
        self.train = train
        self.params = params

        self.encoder_embedding_layer = SeoulEmbeddingLayer(params['hidden_size'])
        self.decoder_embedding_layer = SeoulEmbeddingLayer(params['hidden_size'])
        self.output_embedding_layer = SeoulEmbeddingLayer(params['output_size'])

        self.encoder_stack = EncoderStack(params, train)
        self.decoder_stack = DecoderStack(params, train)

    def __call__(self, inputs, targets):
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            attention_bias = seoul_get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias)

            if targets is None:
                return self.predict(encoder_outputs, attention_bias)
            else:
                outputs = self.decode(targets, encoder_outputs, attention_bias)
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

    def decode(self, targets, encoder_outputs, attention_bias):
        with tf.name_scope("decode"):
            decoder_inputs = self.decoder_embedding_layer(targets)
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.concat(
                    [encoder_outputs[:, -1:, :], decoder_inputs[:, :-1, :]], axis=1)
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

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        pass


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

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes


class GRUAttention(Layer):

    def __init__(self, units, attention_units=None,
                 return_sequences=False, return_state=False,
                 mask_zeros=False, **kargs):
        super(GRUAttention, self).__init__(**kargs)
        self.units = units
        if attention_units is None:
            attention_units = units
        self.attention_units = attention_units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.mask_zeros = mask_zeros
        pass

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        img_features = input_shape[0][-1]
        text_features = input_shape[1][-1]
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        self.kernel = self.add_weight('kernel', shape=(img_features + text_features, 3 * self.units), dtype=dtype)
        self.input_bias = self.add_weight('bias', shape=(3 * self.units,), dtype=dtype)
        self.recurrent_kernel = self.add_weight('recurrent_kernel', shape=(self.units, 3 * self.units), dtype=dtype)
        self.recurrent_bias = self.add_weight('recurrent_bias', shape=(3 * self.units,), dtype=dtype)

        self.att_img_kernel = self.add_weight('att_img_kernel', shape=(img_features, self.attention_units),
                                              dtype=dtype)
        self.att_img_bias = self.add_weight('att_img_bias', shape=(self.attention_units,), dtype=dtype)

        self.att_hidden_kernel = self.add_weight('att_hidden_kernel', shape=(self.units, self.attention_units),
                                                 dtype=dtype)
        self.att_hidden_bias = self.add_weight('att_hidden_bias', shape=(self.attention_units,), dtype=dtype)

        self.att_v_kernel = self.add_weight('att_v_kernel', shape=(self.attention_units, 1),
                                               dtype=dtype)
        self.att_v_bias = self.add_weight('att_v_bias', shape=(1,), dtype=dtype)
        pass

    def get_config(self):
        config = {
            'units': self.units,
            'attention_units': self.attention_units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'mask_zeros': self.mask_zeros
        }
        base_config = super(GRUAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def dense(self, input, kernel, bias):
        return K.dot(input, kernel) + bias

    def attention(self, image, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = K.tanh(self.dense(image, self.att_img_kernel, self.att_img_bias) +
                                        self.dense(hidden_with_time_axis, self.att_hidden_kernel, self.att_hidden_bias))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.dense(attention_hidden_layer, self.att_v_kernel, self.att_v_bias)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = K.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size) ??embedding_dim??
        context_vector = attention_weights * image
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def call(self, input, initial_state=None):
        img, text = input

        def step(cell_inputs, cell_states):
            """Step function that will be used by Keras RNN backend."""
            h_tm1 = cell_states[0]
            features = self.attention(img, h_tm1)
            cell_inputs = K.concatenate([cell_inputs, features], axis=-1)

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(cell_inputs, self.kernel)
            matrix_x = K.bias_add(matrix_x, self.input_bias)

            x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

            # hidden state projected by all gate matrices at once
            matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
            matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)

            recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                                    axis=1)
            z = K.sigmoid(x_z + recurrent_z)
            r = K.sigmoid(x_r + recurrent_r)
            hh = K.tanh(x_h + r * recurrent_h)

            # previous and candidate state mixed by update gate
            h = z * h_tm1 + (1 - z) * hh
            return h, [h]

        if initial_state is None:
            initial_state = (array_ops.zeros((array_ops.shape(text)[0], self.units)),)
        last, sequence, hidden = K.rnn(step, text, initial_state, zero_output_for_mask=self.mask_zeros)
        if self.return_state and self.return_sequences:
            return sequence, hidden
        if self.return_state:
            return last, hidden
        if self.return_sequences:
            return sequence
        return last


if __name__ == '__main__':
    def create_model():
        i1 = tf.keras.layers.Input((64, 200))
        i2 = tf.keras.layers.Input((None, 250))
        g = GRUAttention(128)([i1, i2])

        model = tf.keras.models.Model([i1, i2], g)
        return model

    model = create_model()
    model.summary()
    import numpy as np
    a = np.random.rand(1000, 64, 200)
    b = np.random.rand(1000, 30, 250)
    y = np.random.rand(1000, 128)

    model.compile(loss='mse')
    model.fit([a, b], y, epochs=10)

    print(model.predict([a, b]))
    c = model.predict([a, b])
    model.save('example.hdf5')
    del model
    K.clear_session()
    model = tf.keras.models.load_model('example.hdf5', custom_objects={'GRUAttention': GRUAttention})
    c1 = model.predict([a, b])

    print((c == c1).all())
    print((c == create_model().predict([a, b])).all())

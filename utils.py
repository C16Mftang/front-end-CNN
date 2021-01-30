import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class SlidingWindow(keras.layers.Layer):
    """
    A linear layer calculated with 1-D sliding windows along the first dimension of the input (excluding the batch dimension)

    The input to this layer should have shape (batch_size, seq_len, n_features)

    The output should have shape (batch_size, out_size, n_features)
    """
    def __init__(self, seq_len, n_features, window_size, stride, pads, activation='tanh'):
        super(SlidingWindow, self).__init__()
        self.out_size = int((seq_len-window_size+2*pads)/stride+1)
        self.paddings = tf.constant([[0, 0], [pads, pads], [0, 0]])

        self.seq_len = seq_len
        self.n_features = n_features
        self.window_size = window_size
        self.stride = stride
        self.pads = pads
        self.activation = activation

        self.w = self.add_weight(
            name='w', shape=(self.out_size, 1, window_size), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        # bsz = tf.shape(inputs)[0]
        x = tf.pad(inputs, self.paddings, "CONSTANT")
        # x = tf.stack([tf.slice(x, [0, i, 0], [20, self.window_size, self.n_features]) for i in range(0, self.seq_len-self.window_size+2*self.pads+1, self.stride)])
        x = tf.stack([x[:, i:i+self.window_size, :] for i in range(0, self.seq_len-self.window_size+2*self.pads+1, self.stride)])
        x = tf.transpose(x, perm=[1,0,2,3])
        out = tf.matmul(self.w, x)
        out = tf.squeeze(out, axis=2)
        if self.activation == 'tanh':
            out = tf.math.tanh(out)
        elif self.activation == 'sigmoid':
            out = tf.math.sigmoid(out)
        elif self.activation == 'relu':
            out = tf.nn.relu(out)
        else:
            raise ValueError('Choose one activation from tanh, sigmoid or relu!')
        return out

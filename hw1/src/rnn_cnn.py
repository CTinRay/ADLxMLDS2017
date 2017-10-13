import tensorflow as tf
from tf_classifier_base import TFClassifierBase


class RNNCNNClassifier(TFClassifierBase):
    def _build_model(self):
        placeholders = \
            {'x': tf.placeholder(
                tf.float32,
                shape=(None, None, self._data_shape[2]),
                name='x'),
             'y': tf.placeholder(
                 tf.int32,
                 shape=(None, None),
                 name='y'),
             'training': tf.placeholder(tf.bool, name='training')}

        # calculate sequence length
        lengths = tf.reduce_sum(
            1 - tf.cast(tf.is_nan(placeholders['x'][:, :, 0]),
                        tf.int32),
            axis=-1)

        # mask out those out of length
        x = placeholders['x']
        x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)

        # cnn layer
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1))
        x = tf.layers.conv2d(x, 32, (5, 5), padding='same',
                             activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, (1, 5), (1, 3), padding='same')
        x = tf.reshape(x,
                       (tf.shape(x)[0], tf.shape(x)[1], 69 // 3 * 32))

        # bi-direction cells
        with tf.variable_scope('RNN-Cell-fw'):
            rnn_cell_fw = tf.nn.rnn_cell.GRUCell(100)
        with tf.variable_scope('RNN-Cell-bw'):
            rnn_cell_bw = tf.nn.rnn_cell.GRUCell(100)
        rnn_outputs = \
            tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                            rnn_cell_bw,
                                            x,
                                            sequence_length=lengths,
                                            dtype=tf.float32)[0]
        rnn_outputs = tf.concat(rnn_outputs, axis=-1)

        logits = tf.layers.conv1d(rnn_outputs,
                                  self._n_classes,
                                  1)

        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        # mask out those out of sequence length
        mask = tf.cast(placeholder_y >= 0, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(placeholder_y * mask,
                                                      logits,
                                                      mask)

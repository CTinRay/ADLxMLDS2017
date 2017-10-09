import tensorflow as tf
from tf_classifier_base import TFClassifierBase


class RNNClassifier(TFClassifierBase):
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

        # bi-direction cells
        with tf.variable_scope('RNN-Cell-fw'):
            rnn_cell_fw = tf.nn.rnn_cell.GRUCell(50)
        with tf.variable_scope('RNN-Cell-bw'):
            rnn_cell_bw = tf.nn.rnn_cell.GRUCell(50)

        # calculate sequence length
        lengths = tf.reduce_sum(
            1 - tf.cast(tf.is_nan(placeholders['x'][:, :, 0]),
                        tf.int32),
            axis=-1)

        x = placeholders['x']
        x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
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

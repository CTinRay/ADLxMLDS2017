import tensorflow as tf
from tf_classifier_base import TFClassifierBase


class RNNClassifier(TFClassifierBase):
    def _build_model(self):
        placeholders = \
            {'x': tf.placeholder(
                tf.float32,
                shape=(None, self._data_shape[1], self._data_shape[2]),
                name='x'),
             'y': tf.placeholder(
                 tf.int32,
                 shape=(None, self._data_shape[1]),
                 name='y'),
             'training': tf.placeholder(tf.bool, name='training')}

        # # bi-direction cells        
        # with tf.variable_scope('RNN-Cell-fw'):
        #     rnn_cell_fw = tf.nn.rnn_cell.GRUCell(self._n_classes)
        # with tf.variable_scope('RNN-Cell-bw'):
        #     rnn_cell_bw = tf.nn.rnn_cell.GRUCell(self._n_classes)

        # # calculate the real length of body
        # lengths = tf.reduce_sum(tf.cast(placeholders['y'] > 0, tf.int32),
        #                         axis=-1)

        # rnn_outputs = \
        #     tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
        #                                     rnn_cell_bw,
        #                                     placeholders['x'],
        #                                     sequence_length=lengths,
        #                                     dtype=tf.float32)

        logits = tf.layers.conv1d(placeholders['x'],
                                  self._n_classes,
                                  1)

        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        # mask out those out of sequence length
        mask = tf.cast(placeholder_y >= 0, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(placeholder_y * mask,
                                                      logits,
                                                      mask)

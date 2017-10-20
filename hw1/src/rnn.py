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

        # calculate sequence length
        lengths = tf.reduce_sum(
            1 - tf.cast(tf.is_nan(placeholders['x'][:, :, 0]),
                        tf.int32),
            axis=-1)

        x = placeholders['x']
        x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)

        x = tf.layers.dropout(x, rate=0.2,
                              training=placeholders['training'])
        # bi-direction RNN
        with tf.variable_scope('bi-rnn1'):
            with tf.variable_scope('RNN-Cell-fw'):
                rnn_cell_fw = tf.nn.rnn_cell.GRUCell(256)
            with tf.variable_scope('RNN-Cell-bw'):
                rnn_cell_bw = tf.nn.rnn_cell.GRUCell(256)

            rnn_outputs = \
                tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                rnn_cell_bw,
                                                x,
                                                sequence_length=lengths,
                                                dtype=tf.float32)[0]
            rnn_outputs = tf.concat(rnn_outputs, axis=-1)

        rnn_outputs = tf.layers.dropout(rnn_outputs, rate=0.2,
                                        training=placeholders['training'])
        # bi-direction cells
        with tf.variable_scope('bi-rnn2'):
            with tf.variable_scope('RNN-Cell-fw'):
                rnn_cell_fw = tf.nn.rnn_cell.GRUCell(256)
            with tf.variable_scope('RNN-Cell-bw'):
                rnn_cell_bw = tf.nn.rnn_cell.GRUCell(256)

            rnn_outputs = \
                tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                rnn_cell_bw,
                                                rnn_outputs,
                                                sequence_length=lengths,
                                                dtype=tf.float32)[0]
            rnn_outputs = tf.concat(rnn_outputs, axis=-1)

        rnn_outputs = tf.layers.dropout(rnn_outputs, rate=0.2,
                                        training=placeholders['training'])
        # bi-direction cells
        with tf.variable_scope('bi-rnn3'):
            with tf.variable_scope('RNN-Cell-fw2'):
                rnn_cell_fw = tf.nn.rnn_cell.GRUCell(128)
            with tf.variable_scope('RNN-Cell-bw2'):
                rnn_cell_bw = tf.nn.rnn_cell.GRUCell(128)

            rnn_outputs = \
                tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                rnn_cell_bw,
                                                rnn_outputs,
                                                sequence_length=lengths,
                                                dtype=tf.float32)[0]
            rnn_outputs = tf.concat(rnn_outputs, axis=-1)

        logits = tf.layers.conv1d(rnn_outputs,
                                  self._n_classes + 1,
                                  1)

        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        # mask out those out of sequence length
        mask = tf.cast(placeholder_y >= 0, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(placeholder_y * mask,
                                                      logits,
                                                      mask)
        # lengths = tf.reduce_sum(tf.cast(placeholder_y >= 0, tf.int32), axis=-1)
        # indices = tf.where(placeholder_y >= 0)
        # sparse_y = tf.SparseTensor(indices,
        #                            tf.gather_nd(placeholder_y, indices),
        #                            (self._batch_size, 777))
        # losses = tf.nn.ctc_loss(sparse_y, logits, lengths, time_major=False,
        #                         preprocess_collapse_repeated=True)
        # return tf.reduce_sum(losses)

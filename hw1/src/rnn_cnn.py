import math
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
        x = tf.layers.conv2d(x, 16, (3, 3), padding='same')
        x = tf.layers.batch_normalization(x,
                                          training=placeholders['training'],
                                          name='bn1')
        x = tf.nn.relu(x)
        x1 = tf.layers.max_pooling2d(x, (1, 3), (1, 3), padding='same')

        x2 = tf.layers.conv2d(x1, 32, (3, 3), padding='same')
        x2 = tf.layers.batch_normalization(x2,
                                           training=placeholders['training'],
                                           name='bn2')
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2, 32, (3, 3), padding='same')
        x2 = tf.layers.batch_normalization(x2,
                                           training=placeholders['training'],
                                           name='bn3')
        x2 = tf.nn.relu(x2)
        x2 = tf.concat([x1, x1], axis=-1) + x2
        x2 = tf.layers.max_pooling2d(x2, (1, 3), (1, 3), padding='same')

        x3 = tf.layers.conv2d(x2, 64, (3, 3), padding='same')
        x3 = tf.layers.batch_normalization(x3,
                                           training=placeholders['training'],
                                           name='bn4')
        x3 = tf.nn.relu(x3)
        x3 = tf.layers.conv2d(x3, 64, (3, 3), padding='same')
        x3 = tf.layers.batch_normalization(x3,
                                           training=placeholders['training'],
                                           name='bn5')
        x3 = tf.nn.relu(x3)
        x3 = tf.concat([x2, x2], axis=-1) + x3
        x3 = tf.layers.max_pooling2d(x3, (1, 3), (1, 3), padding='same')

        x_cnn = tf.reshape(x3, (tf.shape(x3)[0], tf.shape(x3)[1],
                                math.ceil(self._data_shape[-1] / 3 / 3 / 3) * 64))

        # bi-direction cells
        with tf.variable_scope('RNN-Cell-fw'):
            rnn_cell_fw = tf.nn.rnn_cell.GRUCell(75)
        with tf.variable_scope('RNN-Cell-bw'):
            rnn_cell_bw = tf.nn.rnn_cell.GRUCell(75)
        rnn_outputs = \
            tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                            rnn_cell_bw,
                                            x_cnn,
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

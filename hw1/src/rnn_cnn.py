import math
import tensorflow as tf
from tf_classifier_base import TFClassifierBase
from weighted_batch_normalization import weighted_batch_normalization


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
        mask = tf.cast(~ tf.is_nan(placeholders['x'][:, :, 0]),
                       tf.int32)
        lengths = tf.reduce_sum(mask, axis=-1)

        # mask out those out of length
        x = placeholders['x']
        x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)

        # cnn layer
        # x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1))

        res_input = tf.layers.conv1d(x, 64, 7,
                                     padding='same')
        res_input = weighted_batch_normalization(
            res_input,
            training=placeholders['training'],
            weights=broadcast(mask, tf.shape(res_input)),
            axis=-1,
            name='bn-init')
        # res_input = tf.layers.batch_normalization(
        #     res_input,
        #     training=placeholders['training'],
        #     name=('bn-init-1'))

        res_input = tf.nn.relu(res_input)

        n_filters = 64
        # resnet
        for l in range(8):
            x = tf.layers.conv1d(res_input, n_filters, 7,
                                 padding='same')
            # x = tf.layers.batch_normalization(
            #     x,
            #     training=placeholders['training'],
            #     name=('bn-%d-1' % l))
            x = weighted_batch_normalization(
                x,
                training=placeholders['training'],
                weights=broadcast(mask, tf.shape(x)),
                name=('bn-%d-1' % l))

            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, 0.1,
                                  training=placeholders['training'])

            x = tf.layers.conv1d(x, n_filters, 7,
                                 padding='same')
            # x = x + tf.concat([res_input, res_input], axis=-1)
            x = x + res_input

            # x = tf.layers.batch_normalization(
            #     x,
            #     training=placeholders['training'],
            #     name=('bn-%d-2' % l))
            x = weighted_batch_normalization(
                x,
                training=placeholders['training'],
                weights=broadcast(mask, tf.shape(x)),
                name='bn-%d-2' % l)

            res_input = tf.nn.relu(x)
            res_input = tf.layers.dropout(x, 0.1,
                                          training=placeholders['training'])
            # n_filters *= 2

        logits = tf.layers.conv1d(res_input,
                                  self._n_classes,
                                  1)
        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        # mask out those out of sequence length
        mask = tf.cast(placeholder_y >= 0, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(placeholder_y * mask,
                                                      logits,
                                                      mask)


def broadcast(x, shape):
    return tf.reshape(tf.cast(x, tf.float32),
                      (tf.shape(x)[0], tf.shape(x)[1], 1)) + tf.zeros(shape)

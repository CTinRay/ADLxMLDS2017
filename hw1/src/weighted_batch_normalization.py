from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.layers import utils
from tensorflow.python.layers.normalization import BatchNormalization


class WeightedBatchNormalization(BatchNormalization):
    """
    Copy most from `tensorflow/tensorflow/python/layers/normalization.py`.
    Moments calculation is modified to accept weights argument.
    """

    def __init__(self, weights=1, **kwargs):
        super(WeightedBatchNormalization, self).__init__(**kwargs)
        self.moments_weights = weights

    def call(self, inputs, training=False):
        if self.fused:
            return self._fused_batch_norm(inputs, training=training)

        # First, compute the axes along which to reduce the mean / variance,
        # as well as the broadcast shape to be used for all parameters.
        input_shape = inputs.get_shape()
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis].value

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        scale, offset = self.gamma, self.beta

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = utils.constant_value(training)
        if training_value is not False:
            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            mean, variance = nn.weighted_moments(inputs, reduction_axes,
                                                 self.moments_weights)
            mean = _smart_select(training,
                                 lambda: mean,
                                 lambda: self.moving_mean)
            variance = _smart_select(training,
                                     lambda: variance,
                                     lambda: self.moving_variance)

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    mean, variance, training)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                scale = array_ops.stop_gradient(r, name='renorm_r')
                offset = array_ops.stop_gradient(d, name='renorm_d')
                if self.gamma is not None:
                    scale *= self.gamma
                    offset *= self.gamma
                if self.beta is not None:
                    offset += self.beta
            else:
                new_mean, new_variance = mean, variance

            # Update moving averages when training, and prevent updates otherwise.
            decay = _smart_select(training, lambda: self.momentum, lambda: 1.)
            mean_update = moving_averages.assign_moving_average(
                self.moving_mean, new_mean, decay, zero_debias=False)
            variance_update = moving_averages.assign_moving_average(
                self.moving_variance, new_variance, decay, zero_debias=False)

            self.add_update(mean_update, inputs=inputs)
            self.add_update(variance_update, inputs=inputs)

        else:
            mean, variance = self.moving_mean, self.moving_variance

        def _broadcast(v):
            if needs_broadcasting and v is not None:
                # In this case we must explicitly broadcast all parameters.
                return array_ops.reshape(v, broadcast_shape)
            return v

        return nn.batch_normalization(inputs,
                                      _broadcast(mean),
                                      _broadcast(variance),
                                      _broadcast(offset),
                                      _broadcast(scale),
                                      self.epsilon)


def weighted_batch_normalization(
        inputs,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer=init_ops.zeros_initializer(),
        gamma_initializer=init_ops.ones_initializer(),
        moving_mean_initializer=init_ops.zeros_initializer(),
        moving_variance_initializer=init_ops.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        training=False,
        trainable=True,
        name=None,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=False,
        weights=1):
    """Functional interface for the batch normalization layer.
    Reference: http://arxiv.org/abs/1502.03167
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    need to be added as a dependency to the `train_op`. For example:
    ```python
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    ```
    Arguments:
      inputs: Tensor input.
      axis: Integer, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      training: Either a Python boolean, or a TensorFlow boolean scalar tensor
        (e.g. a placeholder). Whether to return the output in training mode
        (normalized with statistics of the current batch) or in inference mode
        (normalized with moving statistics). **NOTE**: make sure to set this
        parameter correctly, or else your training/inference will not work
        properly.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
      fused: if `True`, use a faster, fused implementation based on
        nn.fused_batch_norm. If `None`, use the fused implementation if possible.
    Returns:
      Output tensor.
    """
    layer = WeightedBatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        name=name,
        weights=weights,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs, training=training)


def _smart_select(pred, fn_then, fn_else):
    """Selects fn_then() or fn_else() based on the value of pred.
    The purpose of this function is the same as `utils.smart_cond`. However, at
    the moment there is a bug (b/36297356) that seems to kick in only when
    `smart_cond` delegates to `tf.cond`, which sometimes results in the training
    hanging when using parameter servers. This function will output the result
    of `fn_then` or `fn_else` if `pred` is known at graph construction time.
    Otherwise, it will use `tf.where` which will result in some redundant work
    (both branches will be computed but only one selected). However, the tensors
    involved will usually be small (means and variances in batchnorm), so the
    cost will be small and will not be incurred at all if `pred` is a constant.
    Args:
      pred: A boolean scalar `Tensor`.
      fn_then: A callable to use when pred==True.
      fn_else: A callable to use when pred==False.
    Returns:
      A `Tensor` whose value is fn_then() or fn_else() based on the value of pred.
    """
    pred_value = utils.constant_value(pred)
    if pred_value:
        return fn_then()
    elif pred_value is False:
        return fn_else()
    t_then = array_ops.expand_dims(fn_then(), 0)
    t_else = array_ops.expand_dims(fn_else(), 0)
    pred = array_ops.reshape(pred, [1])
    result = array_ops.where(pred, t_then, t_else)
    return array_ops.squeeze(result, [0])

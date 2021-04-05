from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import flatten, cast, get_graph, _is_symbolic_tensor, py_any
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """The class definition for focal loss function to reduce impact of easy instances
    on the overall loss.

    Example:
         ```python
         loss_fn = SparseCategoricalFocalLoss(gamma=2, alpha={0:1,1:3,2:2}, from_logits=True)
         ```
    """
    def __init__(self, gamma=2.0, alpha=None,
                 from_logits=False, **kwargs):
        """Initializes `SparseCategoricalFocalLoss` class.

            Args:
                gamma: exponetial modulating factor.
                alpha: class weights for higher impact of minority class.
                from_logits: whether predictions are logits or not.
                kwargs: other arguments for parent `Loss` class
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def get_config(self):
        """Returns the config dictionary for a `SparseCategoricalFocalLoss` instance."""
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.alpha,
                      from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.

        Returns:
            Loss values in the form of a Tensor
        """
        alpha = self.alpha
        gamma = self.gamma
        from_logits = self.from_logits
        axis = -1

        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)

        probs = y_pred

        # Reformat y_pred shapes
        if (not from_logits and
            not isinstance(y_pred, (ops.EagerTensor, variables_module.Variable)) and
            y_pred.op.type == 'Softmax') and not hasattr(y_pred, '_keras_history'):
            assert len(y_pred.op.inputs) == 1
            y_pred = y_pred.op.inputs[0]
            from_logits = True

        # Clip y_pred to a minimum and maximum value
        if not from_logits:
            epsilon_ = constant_op.constant(K.epsilon(), y_pred.dtype.base_dtype)
            y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1 - epsilon_)
            y_pred = math_ops.log(y_pred)

        # Get dimensions of predictions tensor
        if isinstance(y_pred.shape, (tuple, list)):
            output_rank = len(y_pred.shape)
        else:
            output_rank = y_pred.shape.ndims
        if output_rank is not None:
            axis %= output_rank
            if axis != output_rank - 1:
                permutation = list(
                    itertools.chain(range(axis), range(axis + 1, output_rank), [axis]))
                y_pred = array_ops.transpose(y_pred, perm=permutation)
        elif axis != -1:
            raise ValueError(
                'Cannot compute sparse categorical crossentropy with `axis={}` on an '
                'output tensor with unknown rank'.format(axis))

        # Reformat y_true shape and data type.
        y_true = cast(y_true, 'int64')

        output_shape = array_ops.shape_v2(y_pred)
        target_rank = y_true.shape.ndims

        update_shape = (
                target_rank is not None and output_rank is not None and
                target_rank != output_rank - 1)
        if update_shape:
            y_true = flatten(y_true)
            y_pred = array_ops.reshape(y_pred, [-1, output_shape[-1]])

        # Calculate cross-entropy loss
        if py_any(_is_symbolic_tensor(v) for v in [y_true, y_pred]):
            with get_graph().as_default():
                loss = nn.sparse_softmax_cross_entropy_with_logits_v2(
                    labels=y_true, logits=y_pred)
        else:
            loss = nn.sparse_softmax_cross_entropy_with_logits_v2(
                labels=y_true, logits=y_pred)

        if update_shape and output_rank >= 3:
            loss = array_ops.reshape(loss, output_shape[:-1])

        # Calculate focal modulation to be applied
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        scalar_gamma = gamma.shape.rank == 0

        if alpha is not None:
            alpha = tf.convert_to_tensor(alpha,
                                         dtype=tf.dtypes.float32)
        y_true_rank = y_true.shape.rank
        if not scalar_gamma:
            gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)

        focal_modulation = K.pow(1 - tf.math.reduce_mean(probs, axis=1), gamma)
        focal_modulation = tf.gather(focal_modulation, y_true, axis=0, batch_dims=y_true_rank)

        loss = focal_modulation * loss

        if alpha is not None:
            alpha = tf.gather(alpha, y_true, axis=0,
                              batch_dims=y_true_rank)
            loss *= alpha
        return loss
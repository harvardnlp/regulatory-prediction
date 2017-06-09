from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.gen_nn_ops import *

def atrous_pool2d(value,ksize, rate, padding, name=None, pooling_type="MAX"):
  with ops.name_scope(name, "atrous_pool2d", [value]) as name:
    value = ops.convert_to_tensor(value, name="value")
    if rate < 1:
      raise ValueError("rate {} cannot be less than one".format(rate))

    if rate == 1:
      if pooling_type == "MAX":
        value = nn_ops.max_pool(value=value,
                                ksize=ksize,
                                strides=[1, 1, 1, 1],
                                padding=padding)
        return value
      elif pooling_type == "AVG":
        value = nn_ops.avg_pool(value=value,
                                ksize=ksize,
                                strides=[1, 1, 1, 1],
                                padding=padding)
        return value
      else:
        raise ValueError("Invalid pooling type")


    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      # Handle filters whose shape is unknown during graph creation.
      # if filters.get_shape().is_fully_defined():
      #   filter_shape = filters.get_shape().as_list()
      # else:
      #   filter_shape = array_ops.shape(filters)
      # filter_height, filter_width = filter_shape[0], filter_shape[1]
      kernel_height, kernel_width = ksize[1], ksize[2]


      # Spatial dimensions of the filters and the upsampled filters in which we
      # introduce (rate - 1) zeros between consecutive filter values.
      kernel_height_up = kernel_height + (kernel_height - 1) * (rate - 1)
      kernel_width_up = kernel_width + (kernel_width - 1) * (rate - 1)

      pad_height = kernel_height_up - 1
      pad_width = kernel_width_up - 1

      # When pad_height (pad_width) is odd, we pad more to bottom (right),
      # following the same convention as conv2d().
      pad_top = pad_height // 2
      pad_bottom = pad_height - pad_top
      pad_left = pad_width // 2
      pad_right = pad_width - pad_left
    elif padding == "VALID":
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0
    else:
      raise ValueError("Invalid padding")

    # Handle input whose shape is unknown during graph creation.
    if value.get_shape().is_fully_defined():
      value_shape = value.get_shape().as_list()
    else:
      value_shape = array_ops.shape(value)

    in_height = value_shape[1] + pad_top + pad_bottom
    in_width = value_shape[2] + pad_left + pad_right

    # More padding so that rate divides the height and width of the input.
    pad_bottom_extra = (rate - in_height % rate) % rate
    pad_right_extra = (rate - in_width % rate) % rate

    # The paddings argument to space_to_batch includes both padding components.
    space_to_batch_pad = [[pad_top, pad_bottom + pad_bottom_extra],
                          [pad_left, pad_right + pad_right_extra]]

    value = array_ops.space_to_batch(input=value,
                                     paddings=space_to_batch_pad,
                                     block_size=rate)
    if pooling_type == "MAX":
      value = nn_ops.max_pool(value=value,
                                ksize=ksize,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name=name)

    elif pooling_type == "AVG":
      value = nn_ops.avg_pool(value=value,
                                ksize=ksize,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name=name)
    else:
      raise ValueError("Invalid pooling type")

    # The crops argument to batch_to_space is just the extra padding component.
    batch_to_space_crop = [[0, pad_bottom_extra], [0, pad_right_extra]]

    value = array_ops.batch_to_space(input=value,
                                     crops=batch_to_space_crop,
                                     block_size=rate)

    return value
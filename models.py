from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import ops


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def conv(batch_input, out_channels, stride=2, filter_size=4):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        w = tf.get_variable("kernel", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT") 
        conv = tf.nn.conv2d(input=padded_input, 
                            filter=w, 
                            strides=[1, stride, stride, 1], 
                            padding="VALID") + b
    return conv

def conv_sn(batch_input, out_channels, stride=2, filter_size=4):
    with tf.variable_scope("conv_sn"):
        in_channels = batch_input.get_shape()[3]
        w = tf.get_variable("kernel", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))        
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        conv = tf.nn.conv2d(input=padded_input, 
                            filter=spectral_norm(w), 
                            strides=[1, stride, stride, 1], 
                            padding="VALID") + b
    return conv

def deconv_sn(batch_input, out_channels, stride=2, filter_size=4):
    with tf.variable_scope("conv_sn"):
        in_channels = batch_input.get_shape()[3]
        w = tf.get_variable("kernel", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))        
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(input=batch_input, 
                                      filter=spectral_norm(w), 
                                      output_shape=[batch, in_height * stride, in_width * stride, out_channels], 
                                      strides=[1, stride, stride, 1],
                                      padding="SAME") + b
    return conv

def conv_dialated_sn(batch_input, out_channels, rate=1, filter_size=4):
    with tf.variable_scope("conv_sn"):
        in_channels = batch_input.get_shape()[3]
        w = tf.get_variable("kernel", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))        
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [rate, rate], [rate, rate], [0, 0]], mode="REFLECT")
        conv = tf.nn.atrous_conv2d(input=padded_input, filter=spectral_norm(filter), rate=rate, padding="VALID") + b
    return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        output = (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
    return output

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def spectral_norm(w, iteration=1):
    with tf.variable_scope("spetralnorm"):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

def deconv(batch_input, out_channels, filter_size=4, stride=2):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        w = tf.get_variable("kernel", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))        
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, 
                                      filter=w, 
                                      output_shape=[batch, in_height * stride, in_width * stride, out_channels], 
                                      strides=[1, stride, stride, 1], 
                                      padding="SAME") + b
    return conv

def block_dialated_sn(net, out_channels, dialation_rate=1, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the a resnet block with dialated conv and spectral normalization."""
    with tf.variable_scope(scope, 'BlockDialatedSN', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            skip_connection = net
        with tf.variable_scope('Branch_1'):
            conv = conv_dialated_sn(batch_input=net, out_channels=out_channels, rate=dialation_rate, filter_size=3)
            conv = tf.contrib.layers.instance_norm(conv)
            if activation_fn:
                conv = activation_fn(conv)

            conv = conv_dialated_sn(batch_input=net, out_channels=out_channels, rate=dialation_rate, filter_size=3)
            conv = tf.contrib.layers.instance_norm(conv)
            if activation_fn:
                conv = activation_fn(conv)

        net = scale * conv + skip_connection

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        #if activation_fn:
        #     net = activation_fn(net)
    return net

def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

  return output


def blockA(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the A resnet block."""
  with tf.variable_scope(scope, 'BlockA', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def blockB(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the B resnet block."""
  with tf.variable_scope(scope, 'BlockB', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def blockC(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the C resnet block."""
  with tf.variable_scope(scope, 'BlockC', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net

def mru(x, I, output_channel, stride=1):
    with tf.variable_scope("attention"):
        channel_x = x.shape[3]

        x_I = tf.concat([x, I], axis=3)
        m = slim.conv2d(x_I, channel_x, [3, 3], stride=1,
                               normalizer_fn=None, activation_fn=tf.sigmoid)
        n = slim.conv2d(x_I, output_channel, [3, 3], stride=stride,
                               normalizer_fn=None, activation_fn=tf.sigmoid)

        x_m_I = tf.concat([tf.multiply(x, m), I], axis=3)
        if not stride == 1:
            x = slim.conv2d(x, output_channel, [1, 1], stride=stride,
                               normalizer_fn=None, activation_fn=None)
        z = slim.conv2d(x_m_I, output_channel, [3, 3], stride=stride,
                               normalizer_fn=None, activation_fn=tf.nn.relu)
        y = tf.multiply(x, 1-n) + tf.multiply(z, n)
        
    return y  
    

def demru(x, I, output_channel, stride=1):
    ''' Mask Residual Unit defined in SketchyGAN paper, deconv version '''
    with tf.variable_scope("deattention"):
        channel_x = x.shape[3]

        x_I = tf.concat([x, I], axis=3)
        m = slim.conv2d(x_I, channel_x, [3, 3], stride=1,
                               normalizer_fn=None, activation_fn=tf.sigmoid)
        n = slim.conv2d_transpose(x_I, output_channel, [3, 3], stride=stride,
                               normalizer_fn=None, activation_fn=tf.sigmoid)

        x_m_I = tf.concat([tf.multiply(x, m), I], axis=3)
        if not stride == 1:
            x = slim.conv2d_transpose(x, output_channel, [1, 1], stride=stride,
                               normalizer_fn=None, activation_fn=None)
        z = slim.conv2d_transpose(x_m_I, output_channel, [3, 3], stride=stride,
                               normalizer_fn=None, activation_fn=tf.nn.relu)
        y = tf.multiply(x, 1-n) + tf.multiply(z, n)
        
    return y

def selfatt(input, I, input_channel, flag_I=True, sn=True, channel_fac=16, stride=1):
    ''' Use spectral normalization after every convolution layers '''
    with tf.variable_scope('attention', reuse=False):
        ch = input.get_shape().as_list()[3]
        if flag_I == True:
            x = tf.concat([input, I], axis=3)
        else:
            x = input

        f = ops.conv(x, ch // channel_fac, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
        g = ops.conv(x, ch // channel_fac, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
        h = ops.conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(ops.hw_flatten(g), ops.hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, ops.hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=input.shape) # [bs, h, w, C]
        input = gamma * o + input

    print("Shape of beta...........................................",beta.get_shape(), input.get_shape())
    return input #, beta

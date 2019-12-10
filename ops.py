import tensorflow as tf
import tensorflow.contrib as tf_contrib

import numpy as np
import scipy.io as sio
import scipy.ndimage as sn

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='REFLECT', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'ZERO' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'REFLECT' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

    return x

def conv_dilated(x, channels, kernel=4, rate=1, pad=0, pad_type='REFLECT', use_bias=True, sn=False, scope='conv_dilated_0'):
    with tf.variable_scope(scope):
        if pad_type == 'ZERO' :
            x = tf.pad(x, [[0, 0], [rate, rate], [rate, rate], [0, 0]])
        if pad_type == 'REFLECT' :
            x = tf.pad(x, [[0, 0], [rate, rate], [rate, rate], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.atrous_conv2d(value=x, filters=spectral_norm(w), rate=rate, padding="VALID")
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

    return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

    return x

def upconv(x, channels, kernel=3, stride=2, pad=1, use_bias=True, sn=False, scope='upconv_0'):
    """
    upsampling + conv
    """
    with tf.variable_scope(scope):
        x = up_sample(x, scale_factor=stride)
        x = conv(x, channels=channels, kernel=kernel, stride=1, pad=1, use_bias=use_bias, sn=sn, scope=scope)
    return x

def selfatt(input, condition, input_channel, flag_condition=True, sn=True, channel_fac=16, stride=1, scope='attention_0'):
    ''' Use spectral normalization after every convolution layers '''
    with tf.variable_scope(scope):
        ch = input.get_shape().as_list()[3]
        if flag_condition == True:
            x = tf.concat([input, condition], axis=3)
        else:
            x = input

        f = conv(x, ch // channel_fac, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
        g = conv(x, ch // channel_fac, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
        h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=input.shape) # [bs, h, w, C]
        output = gamma * o + input

    print("Shape of beta...........................................",beta.get_shape(), output.get_shape())
    return output #, beta

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

    return x

def gram_matrix(feature):
    '''
    Comput gram matrix to present style.
    Code borrowed from https://github.com/dongheehand/style-transfer-tf/blob/master/transfer.py
    Defined as Equation 3 in paper https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    '''
    size = tf.shape(feature)
    reshaped_feature_map = tf.reshape(feature, (size[0], size[1] * size[2], size[3]))
    normalization = 2.0 * tf.cast(size[1] * size[2] * size[3] , tf.float32)
    return tf.div(tf.matmul(tf.transpose(reshaped_feature_map, perm = [0,2,1]),reshaped_feature_map) ,normalization)


##################################################################################
# Auxilary Function
##################################################################################
def distance_transform(x):
    # Exact euclidean distance transform. 
    # Equal to 0 if value is 0
    def py_distance_transform(x):
        y = sn.distance_transform_edt(x).astype(np.float32)
        return y
    return tf.py_func(py_distance_transform, [x], tf.float32)

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

    return x + x_init

def resblock_dialated_sn(x_init, channels, kernel=3, rate=1, scale=1.0, use_bias=True, is_training=True, sn=False, reuse=False, scope='resblock'):
    """Builds the a resnet block with dialated conv and spectral normalization."""
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('Branch_0'):
            skip_connection = x_init
        with tf.variable_scope('Branch_1'):
            x = conv_dilated(x_init, channels=channels, kernel=kernel, rate=rate, pad=rate, sn=sn, scope='conv_dilated_0')
            x = tf.contrib.layers.instance_norm(x)
            x = relu(x)

            x = conv_dilated(x, channels=channels, kernel=kernel, rate=rate, pad=rate, sn=sn, scope='conv_dilated_1')
            x = tf.contrib.layers.instance_norm(x)

        net = scale * x + skip_connection

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        #if activation_fn:
        #     net = activation_fn(net)
    return net

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)



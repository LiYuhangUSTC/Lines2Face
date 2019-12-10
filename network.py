from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

from options.train_options import TrainOptions
from models import *
import ops

a = TrainOptions().parse()

EPS = 1e-12

if a.train_stage == 'stage_two' or a.train_stage == 'stage_three':
    a.attention = True
    a.mode = 'train'

##################### Generators #################################################
def create_generator_mru_res(generator_inputs, generator_outputs_channels):
    """
    Replace conv in encoder-decoder network with MRU.
    First and last layer still use conv and deconv.
    No dropout presently.
    Stride = 2, output_channel = input_channel * 2 """
    
    layers = []

    ngf = a.ngf
    with tf.device("/gpu:0"):
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output_e1 = conv(generator_inputs, a.ngf, stride=2)
            rectified = lrelu(output_e1, 0.2)
            layers.append(output_e1)

        with tf.variable_scope("encoder_2"):
            # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            output_e2 = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), ngf * 2, stride=2)
            layers.append(output_e2)

        with tf.variable_scope("encoder_3"):
            # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            output_e3 = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), ngf * 4, stride=2)
            layers.append(output_e3)

        with tf.variable_scope("encoder_4"):
            # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            output_e4 = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), ngf * 6, stride=2)
            layers.append(output_e4)

        with tf.variable_scope("encoder_5"):
            # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            output_e5 = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), ngf * 8, stride=2)
            layers.append(output_e5)

        with tf.variable_scope("encoder_6"):
            # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            output_e6 = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), ngf * 8, stride=2)
            layers.append(output_e6)
    
        with tf.variable_scope("middle"):
            net = layers[-1]
            for i in range(a.num_residual_blocks):
                net = ops.resblock_dialated_sn(net, channels=a.ngf*8, rate=2, sn=a.sn, scope='resblock_%d' % i)
            layers.append(net)

        with tf.variable_scope("decoder_6"):
            # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            input = layers[-1]
            output_d6 = demru(input, tf.image.resize_images(generator_inputs, input.shape[1:3]), ngf * 8, stride=2)
            if a.dropout > 1e-5:
                output_d6 = tf.nn.dropout(output_d6, keep_prob=1 - a.dropout)
            layers.append(output_d6)

        with tf.variable_scope("decoder_5"):
            # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            input = tf.concat([layers[-1], output_e5], axis=3)
            output_d5 = demru(input, tf.image.resize_images(generator_inputs, input.shape[1:3]), ngf * 8, stride=2)
            if a.dropout > 1e-5:
                output_d5 = tf.nn.dropout(output_d5, keep_prob=1 - a.dropout)
            layers.append(output_d5)

        with tf.variable_scope("decoder_4"):
            # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            input = tf.concat([layers[-1], output_e4], axis=3)
            output_d4 = demru(input, tf.image.resize_images(generator_inputs, input.shape[1:3]), ngf * 4, stride=2)
            if a.dropout > 1e-5:
                output_d4 = tf.nn.dropout(output_d4, keep_prob=1 - a.dropout)
            layers.append(output_d4)

        with tf.variable_scope("decoder_3"):
            # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            input = tf.concat([layers[-1], output_e3], axis=3)
            output_d3 = demru(input, tf.image.resize_images(generator_inputs, input.shape[1:3]), ngf * 2, stride=2)
            if a.dropout > 1e-5:
                output_d3 = tf.nn.dropout(output_d3, keep_prob=1 - a.dropout)
            layers.append(output_d3)

        with tf.variable_scope("decoder_2"):
            # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            input = tf.concat([layers[-1], output_e2], axis=3)
            output_d2 = demru(input, tf.image.resize_images(generator_inputs, input.shape[1:3]), ngf * 2, stride=2)
            if a.dropout > 1e-5:
                output_d2 = tf.nn.dropout(output_d2, keep_prob=1 - a.dropout)
            layers.append(output_d2)

    # self-attention layer
    with tf.device("/gpu:%d" % (1)):
        if a.attention:
            with tf.variable_scope("self-attention"): 
                net = layers[-1]
                net = ops.selfatt(net, condition=tf.image.resize_images(generator_inputs, net.get_shape().as_list()[1:3]), 
                                input_channel=a.ngf*2, flag_condition=False, channel_fac=a.channel_fac, scope='attention_0')
                layers.append(net)


        with tf.variable_scope("decoder_1"):
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            input = tf.concat([layers[-1], output_e1], axis=3)
            rectified = tf.nn.relu(input)
            output_d1 = deconv(rectified, generator_outputs_channels)
            output_d1 = tf.tanh(output_d1)
            layers.append(output_d1)

    return layers[-1]

##################### Discriminators ##############################################
def create_discriminator_conv(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []
    
    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv_sn(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv_sn(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers.append(convolved)
        output = tf.sigmoid(convolved)
       
    return [output], [layers]

def create_discriminator_conv_double(discrim_inputs, discrim_targets):
    layers_global = []
    layers_patch = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    ################## shared 
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv_sn(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers_global.append(rectified)
        layers_patch.append(rectified)

    # reused
    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    with tf.variable_scope("layer_2"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_patch.append(rectified)

    # reused
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    with tf.variable_scope("layer_3"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_patch.append(rectified)
        
        share_output = rectified

    ##################### patch branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    with tf.variable_scope("layer_4_patch"):
        convolved = conv_sn(share_output, a.ndf*2, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_patch.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_5_patch"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_patch.append(convolved)
        output_patch = tf.sigmoid(convolved)


    ##################### global branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    with tf.variable_scope("layer_4_global"):
        convolved = conv_sn(share_output, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 16]
    with tf.variable_scope("layer_5_global"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 32]
    with tf.variable_scope("layer_6_global"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_7: [batch, 4, 4, ndf * 32] => [batch, 2,  2,  ndf * 64]
    with tf.variable_scope("layer_7_global"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_8: [batch, 2, 2, ndf * 64] => [batch, 1, 1, 1]
    with tf.variable_scope("layer_8_global"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_global.append(convolved)
        output_global = tf.sigmoid(convolved)
    
    outputs = [output_patch, output_global]
    layers = [layers_patch, layers_global]

    return outputs, layers

def create_discriminator_conv_triple(discrim_inputs, discrim_targets):
    layers_first = []
    layers_second = []
    layers_global = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    ################## shared 
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv_sn(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    with tf.variable_scope("layer_2"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)

    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    with tf.variable_scope("layer_3"):
        convolved = conv_sn(rectified, a.ndf*4, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)
        
        share_output = rectified

    ##################### first branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    with tf.variable_scope("layer_4_first"):
        convolved = conv_sn(share_output, a.ndf*8, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_first.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_5_first"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_first.append(convolved)
        output_first = tf.sigmoid(convolved)

     ##################### second branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    with tf.variable_scope("layer_4_second"):
        convolved = conv_sn(share_output, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_second.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 15,  15,  ndf * 8]
    with tf.variable_scope("layer_5_second"):
        convolved = conv_sn(rectified, a.ndf*8, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_second.append(rectified)

    # layer_6: [batch, 15, 15, ndf * 8] => [batch, 14, 14, 1]
    with tf.variable_scope("layer_6_second"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_second.append(convolved)
        output_second = tf.sigmoid(convolved)

    ##################### last branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    with tf.variable_scope("layer_4_global"):
        convolved = conv_sn(share_output, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 8]
    with tf.variable_scope("layer_5_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 8]
    with tf.variable_scope("layer_6_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_7: [batch, 4, 4, ndf * 32] => [batch, 2,  2,  ndf * 8]
    with tf.variable_scope("layer_7_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_8: [batch, 2, 2, ndf * 64] => [batch, 1, 1, 1]
    with tf.variable_scope("layer_8_global"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_global.append(convolved)
        output_global = tf.sigmoid(convolved)
    
    outputs = [output_first, output_second, output_global]
    layers = [layers_first, layers_second, layers_global]

    return outputs, layers

def create_discriminator_conv_quadruple(discrim_inputs, discrim_targets):
    layers_first = []
    layers_second = []
    layers_third = []
    layers_global = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    ################## shared 
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv_sn(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    with tf.variable_scope("layer_2"):
        convolved = conv_sn(rectified, a.ndf*2, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)

    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    with tf.variable_scope("layer_3"):
        convolved = conv_sn(rectified, a.ndf*4, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)
        layers_first.append(rectified)
        layers_second.append(rectified)
        
        share_output = rectified

    ##################### first branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    with tf.variable_scope("layer_4_first"):
        convolved = conv_sn(share_output, a.ndf*8, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_first.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_5_first"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_first.append(convolved)
        output_first = tf.sigmoid(convolved)

     ##################### second branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    with tf.variable_scope("layer_4_second"):
        convolved = conv_sn(share_output, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_second.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 15,  15,  ndf * 8]
    with tf.variable_scope("layer_5_second"):
        convolved = conv_sn(rectified, a.ndf*8, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_second.append(rectified)

    # layer_6: [batch, 15, 15, ndf * 8] => [batch, 14, 14, 1]
    with tf.variable_scope("layer_6_second"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_second.append(convolved)
        output_second = tf.sigmoid(convolved)

    ##################### third branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    with tf.variable_scope("layer_4_third"):
        convolved = conv_sn(share_output, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_third.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 8]
    with tf.variable_scope("layer_5_third"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_third.append(rectified)

    # layer_6 [batch, 8, 8, ndf * 8] => [batch, 7,  7,  ndf * 8]
    with tf.variable_scope("layer_6_third"):
        convolved = conv_sn(rectified, a.ndf*8, stride=1)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_third.append(rectified)

    # layer_6: [batch, 7, 7, ndf * 8] => [batch, 6, 6, 1]
    with tf.variable_scope("layer_7_third"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_third.append(convolved)
        output_third = tf.sigmoid(convolved)

    ##################### last branch
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    with tf.variable_scope("layer_4_global"):
        convolved = conv_sn(share_output, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 8]
    with tf.variable_scope("layer_5_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 8]
    with tf.variable_scope("layer_6_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_7: [batch, 4, 4, ndf * 32] => [batch, 2,  2,  ndf * 8]
    with tf.variable_scope("layer_7_global"):
        convolved = conv_sn(rectified, a.ndf*8, stride=2)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers_global.append(rectified)

    # layer_8: [batch, 2, 2, ndf * 64] => [batch, 1, 1, 1]
    with tf.variable_scope("layer_8_global"):
        convolved = conv_sn(rectified, out_channels=1, stride=1)
        layers_global.append(convolved)
        output_global = tf.sigmoid(convolved)
    
    outputs = [output_first, output_second, output_third, output_global]
    layers = [layers_first, layers_second, layers_third, layers_global]

    return outputs, layers

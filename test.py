from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import random
import collections
import math
import time

from options.train_options import TrainOptions
import ops
from models import *
from data import *
from network import *

a = TrainOptions().parse()

EPS = 1e-12

NUM_SAVE_IMAGE = 100


Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_fm, gen_loss_style, gen_grads_and_vars, train")
seed = random.randint(0, 2**31 - 1)


##################### Model #######################################################
def create_model(inputs, targets):
    ############### Generator ###########################
    with tf.variable_scope("generator") as scope:
        # float32 for TensorFlow
        inputs = tf.cast(inputs, tf.float32)
        targets = tf.cast(targets, tf.float32)
        out_channels = int(targets.get_shape()[-1])
       
        #Generator
        outputs = create_generator_mru_res(inputs, out_channels)
        beta_list = []

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.device("/gpu:0"):
        # which discriminator
        if a.discriminator == "conv":
            create_discriminator = create_discriminator_conv
        elif a.discriminator == "double":
            create_discriminator = create_discriminator_conv_double
        elif a.discriminator == "triple":
            create_discriminator = create_discriminator_conv_triple
        elif a.discriminator == "quadruple":
            create_discriminator = create_discriminator_conv_quadruple
        
        ############### Discriminator outputs ###########################
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real, feature_real = create_discriminator(inputs, targets)
        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake, feature_fake = create_discriminator(inputs, outputs)


        ################### Loss #########################################
        with tf.name_scope("discriminator_loss"):
            discrim_loss = 0
            for (output_real, output_fake) in zip(predict_real, predict_fake):
                discrim_loss += tf.reduce_mean(-( \
                    tf.log(output_real + EPS) \
                    +  tf.log(1- output_fake + EPS)\
                    ))
            discrim_loss = discrim_loss / len(predict_real)


        ########## Generator loss ##########
        gen_loss = 0.0
        with tf.name_scope("generator_loss"):
            gen_loss_GAN = tf.get_variable("gen_loss_GAN", initializer=tf.constant(0.0))
            for output_fake in predict_fake:
                gen_loss_GAN += tf.reduce_mean(-tf.log(output_fake + EPS))
            gen_loss_GAN = gen_loss_GAN / len(predict_fake)

            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss += gen_loss_GAN * a.gan_weight
            gen_loss += gen_loss_L1 * a.l1_weight

        loss_list = [discrim_loss, gen_loss, gen_loss_GAN, gen_loss_L1]

        with tf.name_scope("gen_fm_loss"):
            if not a.fm:
                gen_loss_fm = 0
            if a.fm:
                gen_loss_fm = tf.get_variable("gen_loss_fm", initializer=tf.constant(0.0))
                for i in range(a.num_feature_matching):
                    gen_loss_fm += tf.reduce_mean(tf.abs(feature_fake[0][-i-1] - feature_real[0][-i-1]))
                gen_loss += gen_loss_fm * a.fm_weight
                loss_list.append(gen_loss_fm)
            
            if not a.style_loss:
                gen_loss_style = 0
            if a.style_loss:
                gen_loss_style = tf.get_variable("gen_loss_style",initializer=tf.constant(0.0))
                for i in range(a.num_style_loss):
                    gen_loss_style += tf.reduce_mean(tf.abs(ops.gram_matrix(feature_fake[0][-i-1]) - ops.gram_matrix(feature_real[0][-i-1])))
                gen_loss += gen_loss_style * a.style_weight
                loss_list.append(gen_loss_style)
                
                
        ################## Restore saver #########################################
        restore_saver = None
        ## restore variables in training stage two
        if a.mode == 'train' and a.train_stage == 'stage_two':
            restore_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_1") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_2") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_3") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_4") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_5") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/encoder_6") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/middle") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_6") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_5") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_4") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_3") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_2") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/decoder_1") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")  
            restore_saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)
        
        ## restore variables in training stage three
        if a.mode == 'train' and a.train_stage == 'stage_three':
            restore_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator") \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
            restore_saver = tf.train.Saver(var_list=restore_var , max_to_keep=1)

        if a.mode == "test":
            # only restore the generator when testing
            restore_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
            restore_saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)

        ################## Train ops #########################################
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr_discrim, a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars, colocate_gradients_with_ops=True)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        
        if a.train_stage == "stage_one" or a.train_stage == "stage_three":
            with tf.name_scope("generator_train"):
                with tf.control_dependencies([discrim_train]):
                    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                    gen_optim = tf.train.AdamOptimizer(a.lr_gen, a.beta1)
                    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars, colocate_gradients_with_ops=True)
                    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        if a.train_stage == "stage_two":
            with tf.name_scope("generator_train"):           
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator/self-attention")] \
                    + [var for var in tf.trainable_variables() if var.name.startswith("generator/decoder_1")]
                gen_optim = tf.train.AdamOptimizer(a.lr_gen, a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars, colocate_gradients_with_ops=True)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
                
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        
        update_losses = ema.apply(loss_list)

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)


    return Model(
        predict_real=predict_real[0],
        predict_fake=predict_fake[0],
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_fm=ema.average(gen_loss_fm),
        gen_loss_style=ema.average(gen_loss_style),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    ), restore_saver

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    mat_dir = os.path.join(a.output_dir, "mats")
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    filesets = []
    if a.load_tfrecord == True:
        for i, fn in enumerate(fetches["filenames"]):
            # int to ascii
            fn = ''.join(chr(n) for n in fn)
            name = fn
            fileset = {"name": name, "step": step}
            if a.mode == 'test':
                for kind in ["inputs", "outputs", "targets"]:
                    filename = name + kind + ".png"
                    out_path = os.path.join(image_dir, filename)
                    contents = fetches[kind][i]
                    # images have been converted to png binary and can be saved by only f.write()
                    with open(out_path, "wb") as f:
                        f.write(contents)

            elif a.mode == 'train':
                for kind in ["outputs"]:
                    filename = name + ".png"
                    out_path = os.path.join(image_dir, filename)
                    contents = fetches[kind][i]
                    # images have been converted to png binary and can be saved by only f.write()
                    with open(out_path, "wb") as f:
                        f.write(contents)
            filesets.append(fileset)
    else:
        for i, in_path in enumerate(fetches["paths"]):
            name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
            fileset = {"name": name, "step": step}
            for kind in ["inputs", "outputs", "targets"]:
                filename = name + "-" + kind + ".png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                fileset[kind] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i]
                with open(out_path, "wb") as f:
                    f.write(contents)
            filesets.append(fileset)
    return filesets

def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for v in fileset:
            index.write("<td><img src='images/%s'></td>" % v)

        index.write("</tr>")
    return index_path

def main():    
    if not os.path.exists(a.output_dir):    
        os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required")
        
    a.scale_size = a.target_size
    a.flip = False
        
    for k, v in a._get_kwargs():
        print(k, "=", v)

    ## read TFRecordDataset    
    examples, iterator  = read_tfrecord()
    print("examples count = %d" % examples.count)

    ## create model
    model, restore_saver = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [a.target_size, int(round(a.target_size * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    ## reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    

    ## only save a part of images for saving driver space.
    num_display_images = 3000
    with tf.name_scope("encode_images"):
        display_fetches = {
            "filenames": examples.filenames,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs[:num_display_images], dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets[:num_display_images], dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs[:num_display_images], dtype=tf.string, name="output_pngs")
        }

    ## summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)
    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    if a.fm:
        tf.summary.scalar("generator_loss_fm", model.gen_loss_fm)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    #    tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
 
    saver = tf.train.Saver(max_to_keep=1)
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, step=step)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
           
    return

main()
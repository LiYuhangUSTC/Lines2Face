from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import random
import math
import collections
import ops

from options.train_options import TrainOptions

a = TrainOptions().parse()
seed = random.randint(0, 2**31 - 1)
Examples = collections.namedtuple("Examples", "filenames, inputs, targets, count, steps_per_epoch")

def transform(image):
    """ Transform image to augment data.
    Including:
        flip: flip image horizontally
        monochrome: rgb to grayscale
        center_crop: center crop image to make sure weight == height
        random_crop: resize image to a larger scale_size and randomly crop it to target a.target_size.
        resize: resize image to [a.scale_size, a.scale_size]        
    """
    # make a copy of image, otherwise get wrong results for unkwon reason
    r = image
    height = r.get_shape()[0] # h, w, c
    width = r.get_shape()[1]
    if a.flip and a.mode == 'train':
        r = tf.image.random_flip_left_right(r, seed=seed)
    if a.monochrome:
        r = tf.image.rgb_to_grayscale(r)
    if not height == width:
        # center crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        r = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
    if  a.random_crop and a.mode == 'train': 
        # resize to a.scale_size and then randomly crop to a.target_size
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        if not a.target_size == a.scale_size:
            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - a.target_size + 1, seed=seed)), dtype=tf.int32)
            if a.scale_size > a.target_size:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.target_size, a.target_size)
            elif a.scale_size < a.target_size:
                raise Exception("scale size cannot be less than crop size") 
    else:
        # resize to a.target_size
        r = tf.image.resize_images(r, [a.target_size, a.target_size], method=tf.image.ResizeMethod.AREA)

    return r 

def parse_function_test(example_proto):        
    '''
     
    '''            
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            'hed': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [512, 512, 3])  
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float32)
    photo = photo * 2. -1.
    height = parsed_features['height']
    width = parsed_features['width']
    depth = parsed_features['depth']
    print(height, width, depth)

    photo = transform(photo)   

    if a.input_type == "df":
        df = tf.decode_raw(parsed_features['df'], tf.float32) 
        df = tf.reshape(df, [512, 512, 1]) 
        if a.df_norm == 'value':# normalize the distance fields, by a given value, to fit grayscale
            df = df / a.df_norm_value
        elif a.df_norm == 'max':# normalize the distance fields, by the max value, to fit grayscale
            df = df / tf.reduce_max(df)
        df = (df) * 2. - 1.    
        df = transform(tf.image.grayscale_to_rgb(df))
        condition = df

    elif a.input_type == "edge": 
        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) 
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = (edge) * 2. - 1.
        edge = transform(tf.image.grayscale_to_rgb(edge))
        condition = edge

    elif a.input_type == "hed": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) 
        hed = tf.reshape(hed, [512, 512, 1])
        hed = (hed) * 2. - 1.
        hed = transform(tf.image.grayscale_to_rgb(hed))
        condition = hed
        
    elif a.input_type == "vg": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) # [0,1], 0~1: probability of being edge
        hed = tf.reshape(hed, [512, 512, 1])

        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) # single pixel edge, 0 or 1, 0:edge, 1:background
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = 1. - edge # 1:edge, 0:background

        vg = tf.multiply(hed, edge) # single pixle probability
        cond = tf.greater(vg, tf.ones(tf.shape(vg)) * a.df_threshold) # thresholding
        vg = tf.where(cond, tf.zeros(tf.shape(vg)), tf.ones(tf.shape(vg))) # single pixle probability after thresholding
        vg = ops.distance_transform(vg)
        vg = tf.reshape(vg, [512, 512, 1])
        if a.df_norm == 'value':
            vg = vg / a.df_norm_value
        elif a.df_norm == 'max':
            vg = vg / tf.reduce_max(vg)
        vg = vg * 2. - 1.
        vg = transform(tf.image.grayscale_to_rgb(vg))
        condition = vg

    return photo, condition, filenames

def parse_function(example_proto):
    '''
     
    '''            
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            'hed': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [512, 512, 3])  
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float32)
    photo = photo * 2. -1.
    height = parsed_features['height']
    width = parsed_features['width']
    depth = parsed_features['depth']
    print(height, width, depth)

    photo = transform(photo)   

    if a.input_type == "df":
        df = tf.decode_raw(parsed_features['df'], tf.float32) 
        df = tf.reshape(df, [512, 512, 1])   
        #df = df/tf.reduce_max(df) # normalize the distance fields, by the max value, to fit grayscale
        df = df / a.df_norm_value # normalize the distance fields, by a given value, to fit grayscale
        df = (df) * 2. - 1.    
        df = transform(tf.image.grayscale_to_rgb(df))
        condition = df

    elif a.input_type == "edge": 
        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) 
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = (edge) * 2. - 1.
        edge = transform(tf.image.grayscale_to_rgb(edge))
        condition = edge

    elif a.input_type == "hed": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) 
        hed = tf.reshape(hed, [512, 512, 1])
        hed = (hed) * 2. - 1.
        hed = transform(tf.image.grayscale_to_rgb(hed))
        condition = hed
        
    elif a.input_type == "vg": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) # [0,1], 0~1: probability of being edge
        hed = tf.reshape(hed, [512, 512, 1])

        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) # single pixel edge, 0 or 1, 0:edge, 1:background
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = 1. - edge # 1:edge, 0:background


        vg = tf.multiply(hed, edge) # single pixle probability
        cond = tf.greater(vg, tf.ones(tf.shape(vg)) * a.df_threshold) # thresholding
        vg = tf.where(cond, tf.zeros(tf.shape(vg)), tf.ones(tf.shape(vg))) # single pixle probability after thresholding
        vg = ops.distance_transform(vg)
        vg = tf.reshape(vg, [512, 512, 1])
        if a.df_norm == 'value':
            vg = vg / a.df_norm_value
        elif a.df_norm == 'max':
            vg = vg / tf.reduce_max(vg)
        vg = vg * 2. - 1.

        vg = transform(tf.image.grayscale_to_rgb(vg))
        condition = vg
    return photo, condition, filenames

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)

    if a.mode=='train':
        dataset = dataset.map(parse_function)   
    else:
        dataset = dataset.map(parse_function_test)  
        
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    if a.mode=='train':
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(a.batch_size)
    iterator = dataset.make_one_shot_iterator()
    photo, condition, filename = iterator.get_next()

    photo.set_shape([a.batch_size, a.target_size, a.target_size, 3])
    condition.set_shape([a.batch_size, a.target_size, a.target_size, 3]) 

    steps_per_epoch = int(math.ceil(a.num_examples / a.batch_size))
    
    return Examples(
        filenames=filename,
        inputs=condition,
        targets=photo,
        count=len(tfrecord_fn),
        steps_per_epoch=steps_per_epoch
    ), iterator

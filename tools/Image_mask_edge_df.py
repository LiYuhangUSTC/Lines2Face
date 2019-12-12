from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import glob
from PIL import Image
import tensorflow as tf
import scipy.io as sio
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--photo_dir", help="path to folder containing photo images")
parser.add_argument("--mask_dir", help="path to folder containing masks")
parser.add_argument("--edge_dir", help="path to folder containing edge maps in mat format")
parser.add_argument("--df_dir", help="path to folder containing distance fields in mat format")
parser.add_argument("--output_dir", default='predict', help="where to put output files")
parser.add_argument("--label_edge", default='predict', help="label of edge maps")
parser.add_argument("--label_df", default='predict', help="label of distance fields")
parser.add_argument("--output_name", required=True, help="name of output file, save as name_train.tfrecords and name_test.tfrecords")
parser.add_argument("--training_fraction", type=float, default=0.7, help="training fraction")
a = parser.parse_args()

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord():
    """
    Read images from photo_dir and mats from mat_dir with label. 
    Save TFRecord file to output_dir.
    
    (training_fraction * photo_num) images and mats will be saved as training data, while the rest
    (1 - training_fraction) * photo_num images and mats will be saved as validation data.
    """
    if a.photo_dir is None or not os.path.exists(a.photo_dir):
        raise Exception("photo_dir does not exist")    
    if a.mask_dir is None or not os.path.exists(a.mask_dir):
        raise Exception("mask_dir does not exist")    
    if a.edge_dir is None or not os.path.exists(a.edge_dir):
        raise Exception("edge_dir does not exist")       
    if a.df_dir is None or not os.path.exists(a.df_dir):
        raise Exception("df_dir does not exist")
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    
    photo_paths = glob.glob(os.path.join(a.photo_dir, "*.jpg"))
    if len(photo_paths) == 0:
        photo_paths = glob.glob(os.path.join(a.photo_dir, "*.png"))
    if len(photo_paths) == 0:
        raise Exception("photo_dir contains no image files")
    print(len(photo_paths))
    
    # for test
    # photo_paths =  photo_paths[:10]
    
    training_num = int(len(photo_paths)*a.training_fraction)  
    f_filenames_path = os.path.join(a.output_dir, a.output_name + "_train_filenames.txt")
    if os.path.exists(f_filenames_path):
        os.remove(f_filenames_path)
    f_filenames = open(f_filenames_path, 'w')    
    num_written = 0
    filename = os.path.join(a.output_dir, a.output_name + '_train.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)    
    for photo_path in tqdm(photo_paths[:training_num]):
        basename, _ = os.path.splitext(os.path.basename(photo_path))
        mask_path = os.path.join(a.mask_dir, basename + ".jpg") 
        edge_path = os.path.join(a.edge_dir, basename + ".mat") 
        df_path = os.path.join(a.df_dir, basename + ".mat") 
        
        try:
            photo_image = Image.open(photo_path)
            photo_array = np.asarray(photo_image, dtype=np.uint8)
        except IOError:
            print("Cannot find file:" + photo_path)
            continue
        try:
            mask_image = Image.open(mask_path)
            mask_array = np.asarray(mask_image, dtype=np.uint8)            
        except IOError:
            print("Cannot find file:" + mask_path)
            continue        
        try:     
            edge_array = sio.loadmat(edge_path)[a.label_edge]
        except IOError:
            print("Cannot find file:" + edge_path)
            continue  
        try:
            df_array = sio.loadmat(df_path)[a.label_df]
        except IOError:
            print("Cannot find file:" + df_path)
            continue
        assert(df_array.shape[0] == photo_array.shape[0] and \
            df_array.shape[1] == photo_array.shape[1])    
    
        height, width = photo_array.shape[:2]
        # depth of photo
        depth = 1 if len(photo_array.shape)==2 else photo_array.shape[2]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': _bytes_feature(basename),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'photo': _bytes_feature(photo_array.tostring()),
            'mask': _bytes_feature(mask_array.tostring()),
            'edge': _bytes_feature(tf.compat.as_bytes(edge_array.tostring())),
            'df': _bytes_feature(tf.compat.as_bytes(df_array.tostring()))          
            }))    
        writer.write(example.SerializeToString())
        f_filenames.write(basename + '\n')
        num_written += 1
        
    f_filenames.write('\n' + str(num_written) + '\n')
    f_filenames.close()    
    writer.close()
    
   
    f_filenames_path = os.path.join(a.output_dir, a.output_name + "_val_filenames.txt")
    if os.path.exists(f_filenames_path):
        os.remove(f_filenames_path)
    f_filenames = open(f_filenames_path, 'w')    
    num_written = 0
    filename = os.path.join(a.output_dir, a.output_name + '_val.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)    
    for photo_path in tqdm(photo_paths[training_num:]):
        basename, _ = os.path.splitext(os.path.basename(photo_path))
        mask_path = os.path.join(a.mask_dir, basename + ".jpg") 
        edge_path = os.path.join(a.edge_dir, basename + ".mat") 
        df_path = os.path.join(a.df_dir, basename + ".mat") 
        
        try:
            photo_image = Image.open(photo_path)
            photo_array = np.asarray(photo_image, dtype=np.uint8)
        except IOError:
            print("Cannot find file:" + photo_path)
            continue
        try:
            mask_image = Image.open(mask_path)
            mask_array = np.asarray(mask_image, dtype=np.uint8)            
        except IOError:
            print("Cannot find file:" + mask_path)
            continue        
        try:     
            edge_array = sio.loadmat(edge_path)[a.label_edge]
        except IOError:
            print("Cannot find file:" + edge_path)
            continue  
        try:
            df_array = sio.loadmat(df_path)[a.label_df]
        except IOError:
            print("Cannot find file:" + df_path)
            continue
        assert(df_array.shape[0] == photo_array.shape[0] and \
            df_array.shape[1] == photo_array.shape[1])    
    
        height, width = photo_array.shape[:2]
        # depth of photo
        depth = 1 if len(photo_array.shape)==2 else photo_array.shape[2]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': _bytes_feature(basename),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'photo': _bytes_feature(photo_array.tostring()),
            'mask': _bytes_feature(mask_array.tostring()),
            'edge': _bytes_feature(tf.compat.as_bytes(edge_array.tostring())),
            'df': _bytes_feature(tf.compat.as_bytes(df_array.tostring()))          
            }))    
        writer.write(example.SerializeToString())
        f_filenames.write(basename + '\n')
        num_written += 1

    f_filenames.write(str(num_written) + '\n')
    f_filenames.close()    
    writer.close()


write_tfrecord()
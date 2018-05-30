# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:26:02 2018

@author: carter
"""

import mxnet as mx
import tensorflow as tf


writer= tf.python_io.TFRecordWriter("C:/PYTHONlibrary/InsightFace_TF-master/datasets/MS1Mface/train.tfrecords") 

imgrec = mx.recordio.MXIndexedRecordIO('train.idx', 'train.rec', 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
imgidx = int(header.label[1]-header.label[0])
print(imgidx)

now_label = 0
i = 1

    
#MXrec2TFrec
while now_label < imgidx:
    for i in range (i,i+10000):
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
            i += 1
writer.close()

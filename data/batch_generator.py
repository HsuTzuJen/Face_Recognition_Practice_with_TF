# -*- coding: utf-8 -*-
"""
Created on Tue May 29 23:14:44 2018

@author: carter
"""

import numpy as np
import multiprocessing as mp
import tensorflow as tf


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))             
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)    
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label      

def cutout(images, length):
    #import concurrent.future.ProcessPoolExcuter as executor
    w = 112
    h = 112
    length = length
    images = images

    for k in range(len(images)):
        x1 = np.random.randint(w-length)
        x2 = x1 + length
        y1 = np.random.randint(h-length)
        y2 = y1 + length
        for i in range(int(x1),int(x2)):
            for j in range(int(y1),int(y2)):
                images[k][i][j][0] = 0.
                images[k][i][j][1] = 0.
                images[k][i][j][2] = 0. 
        x1 = np.random.randint(w-length)
        x2 = x1 + length
        y1 = np.random.randint(h-length)
        y2 = y1 + length
        for i in range(int(x1),int(x2)):
            for j in range(int(y1),int(y2)):
                images[k][i][j][0] = 0.
                images[k][i][j][1] = 0.
                images[k][i][j][2] = 0.    
    return images

class batch_creator:
    def __init__(self, tfrecord_path, shuffle_size, number_process, cutout_size, batch_size=64):
        self.path = tfrecord_path
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.batch_queue = mp.Queue(10)
        self.cutout_size = cutout_size
        self.number_process = number_process
        
        for i in range(self.number_process):
            p = mp.Process(target = batch_creator.create_batch, args = (self,))
            p.start()

    def create_batch(self):
        config = tf.ConfigProto( device_count = {'GPU': 0}    )
        batch_size = 32
        sess = tf.Session(config=config)
        dataset = tf.data.TFRecordDataset(self.path)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()       
        
        for i in range(10000):
            sess.run(iterator.initializer)
            while True:
                try:
                    img_t, lab_t = sess.run(next_element) 
                    for j in range(int(self.batch_size/batch_size)-1):
                        img, lab = sess.run(next_element) 
                        img_t = np.concatenate((img_t, img))
                        lab_t = np.concatenate((lab_t, lab))
                    img_t = cutout(img_t, self.cutout_size)
                except tf.errors.OutOfRangeError:
                    print("End of epoch")
                    break
                self.batch_queue.put([img_t,lab_t])
        
    def get_batch_from_Q(self):
        batch = self.batch_queue.get()
        return batch[0], batch[1]
        
        
if __name__ == '__main__':
    import time
    
    test =  batch_creator('D:/faces/VGG2_2.tfrecords', 3200, 1, 20)    
    img, lab = test.get_batch_from_Q()
    print(img.shape,lab.shape,type(img))
    s = time.time()
    for i in range(100):
        img, lab = test.get_batch_from_Q()
        print(i)      
    e = time.time()
        
    print(100*64/(e-s))
        
        
        

        
        

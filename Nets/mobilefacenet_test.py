# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:25:45 2018

@author: carter
"""


import tensorflow as tf
import tensorlayer as tl

bn_mom = 0.9
#bn_mom = 0.9997

def get_filters(filters,kernel_size, w_init=None , name =None, suffix=''):
    filters = tf.get_variable(name = '%s%s_conv2d/depthwise_kernel' %(name, suffix),shape = [kernel_size[0],kernel_size[1],filters,1], initializer = w_init, dtype = tf.float32)
    return filters


def Conv( inputs, filters=1, kernel_size=(1, 1), strides=(1, 1), padding = 'same', w_init=None, num_group=1, name=None, suffix='', trainable = None):
    if num_group != 1:
        if padding == 'same':
             padding = 'SAME'
        else:
            padding = 'VALID'       
        conv = tf.nn.depthwise_conv2d(inputs, get_filters(filters,kernel_size,w_init=w_init,name=name), strides=[1,strides[0],strides[1],1], padding=padding, name='%s%s_conv2d' %(name, suffix))
    else:
        conv = tf.layers.conv2d ( inputs= inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                  use_bias=False, kernel_initializer=w_init, name='%s%s_conv2d' %(name, suffix))        
    bn = tf.layers.batch_normalization( inputs=conv, name='%s%s_batchnorm' %(name, suffix), momentum=bn_mom, training=trainable,)
    #act = tf.nn.leaky_relu(features=bn, name='%s%s_relu' %(name, suffix))
    #act = tf.nn.selu(features=bn, name='%s%s_selu' %(name, suffix))
    #act = tf.nn.relu( features=bn, name='%s%s_relu' %(name, suffix))
    bn = tl.layers.InputLayer(bn)
    act = tl.layers.PReluLayer(bn, name='%s%s_Prelu' %(name, suffix))
    
    return act.outputs
    
    return act
    
def Linear( inputs, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', w_init=None, num_group=1, name=None, suffix='', trainable = None):
    if num_group != 1:
        conv = tf.layers.separable_conv2d(inputs= inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                          use_bias=False, depthwise_initializer=w_init, pointwise_initializer=w_init, name='%s%s_conv2d' %(name, suffix))
    else:
        conv = tf.layers.conv2d ( inputs= inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                  use_bias=False, kernel_initializer=w_init, name='%s%s_conv2d' %(name, suffix))        
    bn = tf.layers.batch_normalization( inputs=conv, name='%s%s_batchnorm' %(name, suffix), momentum=bn_mom, training=trainable,)  
    return bn

    
def DResidual( inputs, num_out=1, kernel_size=(3, 3), strides=(2, 2), padding='same', num_group=1, name=None, w_init=None, suffix='', trainable = None):
    #conv = Conv( inputs= inputs, filters=num_group, kernel_size=(1, 1), padding='valid', strides=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv = tf.layers.separable_conv2d(inputs=inputs, filters=num_group, kernel_size=(1, 1), padding='valid', strides=(1, 1), 
                                                use_bias=False, depthwise_initializer=w_init, pointwise_initializer=w_init, name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv( inputs=conv, filters=num_group, num_group=num_group, kernel_size=kernel_size, padding=padding, strides=strides, name='%s%s_conv_dw' %(name, suffix), w_init=w_init, trainable=trainable)
    proj = Linear( inputs=conv_dw, filters=num_out, kernel_size=(1, 1), padding='valid', strides=(1, 1), name='%s%s_conv_proj' %(name, suffix), trainable=trainable)
    return proj
    
def Residual( inputs, num_block=1, num_out=1, kernel_size=(3, 3), strides=(1, 1), padding='same', num_group=1, name=None, w_init=None, suffix='', trainable = None):
    identity= inputs
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual( inputs=identity, num_out=num_out, kernel_size=kernel_size, strides=strides, padding=padding, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i, w_init=w_init, trainable=trainable)
    	identity=conv+shortcut
    return identity
        

def get_symbol(inputs, w_init=None, reuse=False, scope=None, trainable = None):
    global bn_mom
    bn_mom =  0.9
    #wd_mult =  1.
    with tf.variable_scope(scope, reuse=reuse):
        conv_1 = Conv( inputs, filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), name="conv_1", w_init=w_init, trainable=trainable)
        conv_2_dw = Conv(conv_1, num_group=64, filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), name="conv_2_dw", w_init=w_init, trainable=trainable)
        conv_23 = DResidual(conv_2_dw,          num_out=64, kernel_size=(3, 3), strides=(2, 2), padding='same', num_group=128, name="dconv_23", w_init=w_init, trainable=trainable)
        conv_3 = Residual(conv_23, num_block=4, num_out=64, kernel_size=(3, 3), strides=(1, 1), padding='same', num_group=128, name="res_3", w_init=w_init, trainable=trainable)
        conv_34 = DResidual(conv_3,             num_out=128, kernel_size=(3, 3), strides=(2, 2), padding='same', num_group=512, name="dconv_34", w_init=w_init, trainable=trainable)
        conv_4 = Residual(conv_34, num_block=6, num_out=128, kernel_size=(3, 3), strides=(1, 1), padding='same', num_group=256, name="res_4", w_init=w_init, trainable=trainable)
        conv_45 = DResidual(conv_4,             num_out=128, kernel_size=(3, 3), strides=(2, 2), padding='same', num_group=512, name="dconv_45", w_init=w_init, trainable=trainable)
        conv_5 = Residual(conv_45, num_block=2, num_out=128, kernel_size=(3, 3), strides=(1, 1), padding='same', num_group=256, name="res_5", w_init=w_init, trainable=trainable)   
        conv_6_sep = tf.layers.separable_conv2d(inputs=conv_5, filters=512, kernel_size=(1, 1), padding='valid', strides=(1, 1), 
                                                use_bias=False, depthwise_initializer=w_init, pointwise_initializer=w_init, name="conv_6sep")
        conv_6_dw = Conv(conv_6_sep, filters=512, num_group=512, kernel_size=(7,7), padding='valid', strides=(1, 1), name="conv_6dw7_7", w_init=w_init, trainable=trainable)  
    #conv_6_dw = mx.symbol.Dropout( inputs=conv_6_dw, p=0.4)
        #net_shape = net.outputs.get_shape()
        net = tf.reshape(conv_6_dw, shape=[-1, 512], name='reshape_conv_6dw7_7')
        net = tf.layers.dense(net, units=128, kernel_initializer=w_init, name='E_DenseLayer')
        net = tf.layers.batch_normalization( inputs=net, epsilon=2e-5, momentum=bn_mom, training=trainable, name='fc1')
        net = tl.layers.InputLayer(net)
        return net

if __name__ == '__main__':
        config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )    
        x = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_place')
        sess = tf.Session(config = config)
        # w_init = tf.truncated_normal_initializer(mean=10, stddev=5e-2)
        w_init = tf.contrib.layers.xavier_initializer(uniform=True)
        # test resnetse
        nets = get_symbol(inputs=x, w_init=w_init, reuse=tf.AUTO_REUSE, scope='mobilefacenet')
        
        from functools import reduce
        from operator import mul

        with sess:
            sess.run(tf.global_variables_initializer())
            num_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                num_params += reduce(mul, [dim.value for dim in shape], 1)
                if 'gamma' not in variable.name and 'beta' not in variable.name and 'alphas' not in variable.name :
                    print(variable.name,variable.shape)
            print(num_params)

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            saver.save(sess, './test.ckpt')



# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 04:10:33 2018

@author: carter
"""
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import argparse
import os
from losses.face_losses import arcface_loss
from losses.face_losses import LMCL
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test
from Nets.mobilefacenet_test import get_symbol
from data.batch_generator import batch_creator


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[80000, 100000,120000, 140000,160000,180000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_vgg_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/faces_ms1m_112x112', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=2000, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=20000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--cutout_length', default=20, help='cutout')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    
    trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    get_batch = batch_creator(tfrecord_path = 'C:/MS1Mtrain.tfrecords', shuffle_size = 19200, number_process = 1, cutout_size = args.cutout_length, batch_size=args.batch_size)
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db) 
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_symbol(inputs=images, w_init=w_init_method, reuse=False, scope='mobilefacenet', trainable = trainable)
    # 3.10 define sess
    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)#, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    '''
    sess.run(tf.global_variables_initializer())
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, './output/ckpt/LIR101+cutout_iter_244000_acc_0.9921666666666666.ckpt')    
    '''
    # 3.2 get arcface loss
    margin = tf.placeholder(tf.float32, name='margin')
    
    now_m ,logit = LMCL(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output, m = margin, s=128.)#, reuse=False)
    # test net  because of batch normal layer
    #tl.layers.set_name_reuse(True)
    #test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=True)
    embedding_tensor = net.outputs
    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    # inference_loss_avg = tf.reduce_mean(inference_loss)
    # 3.4 define weight deacy losses    
    wd_loss = 0
    
    for variable in tf.trainable_variables():
        if 'bias' in variable.name :
            wd = args.weight_deacy
        else:
            wd = args.weight_deacy
            print(variable.name,'use wd =',wd)
            wd_loss += tf.contrib.layers.l2_regularizer(wd)(variable)

    
    # 3.5 total losses
    total_loss = inference_loss + wd_loss
    
    model_name = 'agedbmobileDM(s=128from1080K)b{}.log'.format(args.batch_size)
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/'+ model_name
    log_file = open(log_file_path, 'a')
    # 4 begin iteration
    count = 1364000
    count_add = int(args.batch_size/64)
    total_accuracy = {}
    acc_save = 0.9306666666666665
    max_acc_iter = count
    lowest_inference_loss_val = 2.5
    lowest_acc_val = 0.7
    now_acc = 0.9306666666666665
    use_m = 0.5
    fix_m = True    
    
    # 3.6 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [int(p*val-count) for val in args.lr_steps]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001,0.00001,0.00001,0.00001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)  
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    #saver.save(sess,'./output/ckpt/test.ckpt')
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, './output/ckpt/agedbmobileDM(s=128from1080K)b64.log_iter_1364000_acc_0.9306666666666665.ckpt')
    
    #initailzie uninitialized weights
    '''
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    sess.run(tf.variables_initializer(uninitialized_vars))
    '''   
    # 4 begin iteration
    for i in range(args.epoch):
        #break
        #sess.run(iterator.initializer)
        while True:
            try:
                start = time.time()
                images_train, labels_train = get_batch.get_batch_from_Q()
                
                feed_dict = {images: images_train, labels: labels_train, trainable: True, margin:use_m}
                feed_dict.update(net.all_drop)
                
                test_now_m = sess.run(now_m,feed_dict=feed_dict)
                test_now_m = np.mean(test_now_m)
                use_m = test_now_m
                
                _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val, test_now_m = \
                    sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc, now_m],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                test_now_m = np.mean(test_now_m)
                if fix_m == True:
                    use_m = test_now_m
                    use_m =int(use_m*100)/100.0
                    #fix_m = False
                    
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information

                if count > 0 and count % args.show_info_interval == 0 or inference_loss_val < lowest_inference_loss_val or acc_val>lowest_acc_val:
                    print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                          'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec),
                          '\nnow max acc =',now_acc,'iter =', max_acc_iter, '\nusing m ={}, max(W_i, W_j) ={}'.format(use_m, 1-test_now_m))

                count += count_add


                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, margin: use_m, trainable: True}
                    feed_dict.update(net.all_drop)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)
                # validate
                if count > 0 and count % args.validate_interval == 0 or inference_loss_val < lowest_inference_loss_val or acc_val>lowest_acc_val:
                    feed_dict_test ={trainable: False}
                    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    print('test accuracy is: ', str(results[0]))
                    total_accuracy[str(count)] = results[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    max_result = max(results)
                    if max_result > now_acc:
                        now_acc = max_result
                        use_m = test_now_m
                        use_m =int(use_m*100)/100.0
                    
                    if max_result > acc_save :
                        max_acc_iter = count
                        acc_save = max_result
                        print('best accuracy is %.5f' % acc_save)
                        filename = '{}_iter_{}_acc_{}'.format(model_name,count,acc_save) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max_result)+'\n')
                        log_file.write(filename+'\n')
                        log_file.flush()
                if inference_loss_val < lowest_inference_loss_val:
                    lowest_inference_loss_val = inference_loss_val
                if acc_val>lowest_acc_val :
                    lowest_acc_val = acc_val

                 #save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = '{}_iter_{}_acc_{}'.format(model_name,count,max_result) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
                    
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
    log_file.close()
    log_file.write('\n')

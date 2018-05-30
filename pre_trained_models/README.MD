
Load pretrained model as follows:

net = get_symbol(inputs=images, w_init=None, reuse=tf.AUTO_REUSE, scope='mobilefacenet', trainable = False)
restore_saver = tf.train.Saver()
restore_saver.restore(sess,'model_path')

Create batches and put them in a mutilprocessing.Queue.

Do not have to wait sess.run(next_element) in your training process.
It makes your training process faster.


Cutout method has been used in image preprocessing. Paper: https://arxiv.org/abs/1708.04552

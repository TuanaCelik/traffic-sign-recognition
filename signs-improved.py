############################################################
#                                                          #
#  Code for Traffic-sign-recognition coursework            #
#  Based on the "A Shallow Network with Combined Pooling   #
#  for Fast Traffic Sign Recognition" paper by Jianming    #
#  Zhang, Qianqian Huang , Honglin Wu and Yukai Liu.       #
#                                                          #
############################################################
from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf

import cPickle as pickle
from batch_generator import batch_generator

here = os.path.dirname(__file__)
sys.path.append(here)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('epochs', 45, 'Number of epochs')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch.')
tf.app.flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay factor.')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate') 
run_log_dir = os.path.join(FLAGS.log_dir,
                           ('log_replica_normalization_withening').format(wd=FLAGS.weight_decay, lr=FLAGS.learning_rate))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)



def deepnn(x_image, class_count=43):
    
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    
    #x_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)

    padded_input = tf.pad(x_image, [[0, 0],[2, 2], [2, 2], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv2d(
        inputs=padded_input,
        filters=32,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv1'
    )
    #tf.summary.merge([tf.summary.image('Kernel after conv 1', conv1)])
    pool1 = tf.layers.average_pooling2d(
        inputs=tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        pool_size=[3, 3],
        strides=2,
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=tf.pad(pool1,[[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT"),
        filters=32,
        kernel_size=[5, 5], 
        padding='valid',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv2'
    )

    pool2 = tf.layers.average_pooling2d(
        inputs=tf.pad(conv2, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        pool_size=[3, 3],
        strides=2,
        name='pool2'
    )

    conv3 = tf.layers.conv2d(
        inputs=tf.pad(pool2,[[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT"),
        filters=64,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv3'
    )


    pool3 = tf.layers.max_pooling2d(
        inputs=tf.pad(conv3, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        pool_size=[3, 3],
        strides=2,
        name='pool3'
    )
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv4'
    )
    conv4 = tf.contrib.layers.flatten(conv4)
    logits = tf.contrib.layers.fully_connected(
            inputs=conv4,
            num_outputs=class_count,
            activation_fn=None,
            weights_initializer=initializer,
            weights_regularizer=regularizer
        )

    return logits

mean_channel = [0,0,0]
stddev_channel = [0,0,0]

def whiten(data):
    print("whitening training images")
    data = np.array(data)
    for i in range(0,3):
        mean_channel[i] = np.mean(data[:,0][:][:][i])
        stddev_channel[i] = np.std(data[:,0][:][:][i])
        data[:,0][:][:][i] = (data[:,0][:][:][i]  - mean_channel[i]) / stddev_channel[i]
    return data

def whiten_test(data):
    print("whitening test images")
    data = np.array(data)
    for i in range(0,3) :
        data[:,0][:][:][i] = (data[:,0][:][:][i]  - mean_channel[i]) / stddev_channel[i]
    return data

def preprocess(image, channels=3): 
    for i in range(0,3) : 
        mean_c = np.mean(image[0][:, :, i]) 
        stddev_c = np.std(image[0][:, :, i]) 
        image[0][:, :, i] = (image[0][:, :, i]  - mean_c) / stddev_c 
    return image

def put_kernels_on_grid(kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

def main(_):
    tf.reset_default_graph()

    dataset = pickle.load(open('dataset.pkl', 'rb'))

    print("Per-image normalization..")
    trainData = dataset[0]
    testData = dataset[1]

    # trainData = whiten(dataset[0])
    trainD = map(lambda img: preprocess(img), dataset[0])
    trainData = whiten(trainD)

    # testData = whiten_test(dataset[1])
    testD = map(lambda img: preprocess(img), dataset[1])

    # ---- Per class accuracy 
    # filterTest = list(filter(lambda x: ((np.argmax(x[1]) > 11 and np.argmax(x[1]) < 15) or np.argmax(x[1]) == 17) , dataset[1]))
    # print(len(filterTest))
    # testD = map(lambda img: preprocess(img), filterTest)
    # ----
    testData = whiten_test(testD)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
        x_image= tf.reshape(x, [-1, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 43])

    with tf.variable_scope('model'):
        logits = deepnn(x_image)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='error')
        
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(cross_entropy)

    with tf.variable_scope('conv1'):
        # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')
        # print(vars)
        tf.get_variable_scope().reuse_variables()
        kernels1 = tf.get_variable('kernel')
        grid1 = put_kernels_on_grid (kernels1)
        kernel_summary_c1 = tf.image.summary('conv1/kernel', grid1, max_outputs=1)        


    with tf.variable_scope('conv2'):
        tf.get_variable_scope().reuse_variables()
        kernels2 = tf.get_variable('kernel')
        reduceK = tf.reduce_mean(kernels2,axis=2,keep_dims=True)
        grid2 = put_kernels_on_grid (reduceK)
        kernel_summary_c2 = tf.image.summary('conv2/kernel', grid2, max_outputs=1)     

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Error", accuracy)
    img_summary = tf.summary.image('Input Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, img_summary, kernel_summary_c1, kernel_summary_c2])

    
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        #declare writers
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        #initialize the variables
        sess.run(tf.global_variables_initializer())
        f = open('replica.txt', 'w')
        print("Training....")
        for step in range(0, FLAGS.epochs, 1):
            
            train_batch_generator = batch_generator(trainData, 'train', batch_size=FLAGS.batch_size)
            print("Epoch: {}".format(step))


            for (train_images, train_labels) in train_batch_generator:
                 _, train_summary_str = sess.run([train_step, train_summary],
                                                 feed_dict={x_image: train_images, y_: train_labels})

            
            train_writer.add_summary(train_summary_str, step) 
                                                                        
            train_writer.flush()
      

        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        print("Testing...")
        test_batch_generator = batch_generator(testData, 'test', batch_size=FLAGS.batch_size)

        for (test_images, test_labels) in test_batch_generator:
            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: test_images, y_: test_labels})
            print('Batch {}, accuracy : {}'.format(batch_count, test_accuracy_temp))

            test_accuracy += test_accuracy_temp

            test_summay_str = sess.run(img_summary, feed_dict={x_image: test_images})
            test_writer.add_summary(test_summay_str, evaluated_images)
            
            batch_count += 1

       

        test_accuracy = 100 * (test_accuracy / batch_count)
        print('Accuracy on test set: %0.3f%%' % test_accuracy)
        print('model saved to ' + checkpoint_path)
        f.write('Accuracy on test set: %0.3f%%' % test_accuracy)
        train_writer.close()
        test_writer.close()
        f.close()
   
if __name__ == '__main__':
    tf.app.run(main=main)
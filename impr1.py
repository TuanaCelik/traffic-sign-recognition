# Changed activation function to crelu -  improved up to 95%, if the epochs was 50 it went lower,
# but this both needs further testing ( Could run it a few times and take an average with them)


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
tf.app.flags.DEFINE_float('weight-decay', 1e-4, 'Weight decay factor.')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Learning rate') 
run_log_dir = os.path.join(FLAGS.log_dir,
                           ('log_wd_{weight-decay}_lr_{learning_rate}_'))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)



def deepnn(x_image, class_count=43):
    
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    
    x_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)

    padded_input = tf.pad(x_image, [[0, 0],[2, 2], [2, 2], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv2d(
        inputs=padded_input,
        filters=32,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv1'
    )
    
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
        activation=tf.nn.crelu,
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
        activation=tf.nn.crelu,
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
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv4'
    )
   
    logits = tf.contrib.layers.fully_connected(
            inputs=conv4,
            num_outputs=class_count,
            activation_fn=None,
            weights_initializer=initializer,
            weights_regularizer=regularizer
        )

    logits = tf.contrib.layers.flatten(logits)   
    return logits


def main(_):
    tf.reset_default_graph()

    dataset = pickle.load(open('dataset.pkl', 'rb'))

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
        x_image= tf.reshape(x, [-1, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 43])

    with tf.variable_scope('model'):
        logits = deepnn(x_image)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(cross_entropy)

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    img_summary = tf.summary.image('Input Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, img_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        #declare writers
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        #initialize the variables
        sess.run(tf.global_variables_initializer())
        print("Training....")
        for step in range(0, FLAGS.epochs, 1):
            print("Epoch: {}".format(step+1))

            train_batch_generator = batch_generator(dataset, 'train', batch_size=FLAGS.batch_size)

            for (train_images, train_labels) in train_batch_generator:
                 _, train_summary_str = sess.run([train_step, train_summary],
                                                 feed_dict={x_image: train_images, y_: train_labels})

            
            train_writer.add_summary(train_summary_str, step) 
                                                                        
            train_writer.flush()
      

        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        print("Testing...")
        test_batch_generator = batch_generator(dataset, 'test', batch_size=FLAGS.batch_size)

        for (test_images, test_labels) in test_batch_generator:
            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: test_images, y_: test_labels})
            print('Batch {}, accuracy : {}'.format(batch_count, test_accuracy_temp))

            test_accuracy += test_accuracy_temp

            test_summay_str = sess.run(img_summary, feed_dict={x_image: test_images})
            test_writer.add_summary(test_summay_str, evaluated_images)
            
            batch_count += 1

       

        test_accuracy = 100 * (test_accuracy / batch_count)
        print('Accuracy on test set with improvements: %0.3f%%' % test_accuracy)
        print('model saved to ' + checkpoint_path)

        train_writer.close()
        test_writer.close()
       
   
if __name__ == '__main__':
    tf.app.run(main=main)

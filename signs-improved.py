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
tf.app.flags.DEFINE_integer('epochs', 50, 'Number of epochs')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch.')
tf.app.flags.DEFINE_float('weight_decay', 1e-6, 'Weight decay factor.')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate') 
run_log_dir = os.path.join(FLAGS.log_dir,
                           ('log_with_all_improvements').format(ep=FLAGS.epochs, wd=FLAGS.weight_decay, lr=FLAGS.learning_rate))

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
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='conv1'
    )
    conv1 = tf.nn.crelu(tf.layers.batch_normalization(conv1, name='conv1'))
    pool1 = tf.layers.max_pooling2d(
        inputs=tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=2,
        name='pool1'
    )
    pool1_multi = tf.layers.max_pooling2d(
        inputs=tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=8,
        name='pool1_multi'
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
    conv2 = tf.nn.crelu(tf.layers.batch_normalization(conv2, name='conv2'))
    pool2 = tf.layers.max_pooling2d(
        inputs=tf.pad(conv2, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=2,
        name='pool2'
    )
    pool2_multi = tf.layers.max_pooling2d(
        inputs=tf.pad(conv2, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=4,
        name='pool2_multi'
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
    conv3 = tf.nn.crelu(tf.layers.batch_normalization(conv3, name='conv3'))
    pool3 = tf.layers.max_pooling2d(
        inputs=tf.pad(conv3, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=2,
        name='pool3'
    )
    pool3_multi = tf.layers.max_pooling2d(
        inputs=tf.pad(conv3, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
        padding='valid',
        pool_size=[3, 3],
        strides=2,
        name='pool3_multi'
    )
    #pool3 = tf.nn.dropout(pool3, 0.7)
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
    conv4 = tf.nn.crelu(tf.layers.batch_normalization(conv4, name='conv4'))
    pool1_multi = tf.contrib.layers.flatten(pool1_multi)
    pool2_multi = tf.contrib.layers.flatten(pool2_multi)
    pool3_multi = tf.contrib.layers.flatten(pool3_multi)
    conv4 = tf.contrib.layers.flatten(conv4)
    conv4 = tf.concat([pool1_multi, pool2_multi, pool3_multi, conv4], axis=1)
    conv4 = tf.nn.dropout(conv4, 0.7)
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
    print("normalizing test images")
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

def main(_):
    tf.reset_default_graph()
    dataset = pickle.load(open('dataset.pkl', 'rb'))
    
    print("Per-image normalization..")
    trainData = map(lambda img: preprocess(img), dataset[0])
    trainData = whiten(trainD)

    testData = map(lambda img: preprocess(img), dataset[1])
    testData = whiten_test(testD)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
        x_image= tf.reshape(x, [-1, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 43])
        global_epoch = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        
        augment = tf.placeholder(tf.bool)        
        rotated = tf.map_fn(lambda img: tf.contrib.image.rotate(img, tf.random_uniform([],-0.26,0.26)), x_image)
        translated = tf.map_fn(lambda img: tf.contrib.image.transform(img,[1, 0, tf.random_uniform([],-2,2), 0,1, tf.random_uniform([],-2,2), 0, 0]), x_image)
        
        x_image = tf.cond(augment,  lambda: tf.concat([x_image, rotated, translated], axis=0), lambda: tf.identity(x_image))
        y_ = tf.cond(augment, lambda: tf.concat([y_, y_], axis=0), lambda: tf.identity(y_))

    with tf.variable_scope('model'):
        logits = deepnn(x_image)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        
        decay_steps = 10
        decay_rate = 0.95  
        global_step = tf.Variable(0, trainable=False)

        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_epoch, decay_steps, decay_rate, staircase=False)
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate, 0.9).minimize(cross_entropy)

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    img_summary = tf.summary.image('Input Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, img_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    validation_temp = 0
    patience = 1
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        best_saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        #declare writers
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        test_batch_gen = batch_generator(testData, 'test', batch_size=FLAGS.batch_size)
        
        sess.run(tf.global_variables_initializer())
        f = open('improvements.txt', 'w')
        print("Training....")
        for step in range(0, FLAGS.epochs, 1):
            
            train_batch_generator = batch_generator(trainData, 'train', batch_size=FLAGS.batch_size)


            for (train_images, train_labels) in train_batch_generator:
                 _, train_summary_str = sess.run([train_step, train_summary],
                                                 feed_dict={x_image: train_images, y_: train_labels, global_epoch: step, augment: True})


            test_accuracy = 0
            batch_count = 0
            test_batch_generator = batch_generator(testData, 'test', batch_size=FLAGS.batch_size)

            for (test_images, test_labels) in test_batch_generator:
                test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: test_images, y_: test_labels, augment: False})
                test_accuracy += test_accuracy_temp
                batch_count += 1

            validtion = test_accuracy / batch_count

            print("Epoch: {}".format(step+1) + " Validation Accuracy: {}".format(validtion))
            train_writer.add_summary(train_summary_str, step)
            if step % 10 == 0:
                train_writer.flush()

            if(validtion > validation_temp):
                patience = 1
                #train_writer.add_summary(train_summary_str, step)
                validation_temp = validtion
                best_saver.save(sess, 'best.ckpt')
                train_writer.flush()
            else :
                patience += 1
            if (patience == 10):
                best_saver.restore(sess, 'best.ckpt')
                break
            #train_writer.flush()
                                                                        
      

        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        print("Testing...")
        test_batch_generator = batch_generator(testData, 'test', batch_size=FLAGS.batch_size)

        for (test_images, test_labels) in test_batch_generator:
            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: test_images, y_: test_labels, augment: False})
            print('Batch {}, accuracy : {}'.format(batch_count, test_accuracy_temp))
            #f.write('Batch {}, accuracy : {}'.format(batch_count, test_accuracy_temp))
            test_accuracy += test_accuracy_temp

            test_summay_str = sess.run(img_summary, feed_dict={x_image: test_images, augment: False})
            test_writer.add_summary(test_summay_str, evaluated_images)
            
            batch_count += 1

       

        test_accuracy = 100 * (test_accuracy / batch_count)
        print('Accuracy on test set with improvements: %0.3f%%' % test_accuracy)
        f.write('Accuracy on test set with improvements: %0.3f%%' % test_accuracy)
        print('model saved to ' + checkpoint_path)
        train_writer.close()
        test_writer.close()
        f.close()
   
if __name__ == '__main__':
    tf.app.run(main=main)
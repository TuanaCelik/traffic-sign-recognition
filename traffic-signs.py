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

from gtsrb import *
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper

here = os.path.dirname(__file__)
sys.path.append(here)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries.' +
                            ' (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 1000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Number of examples to run. (default: %(default)d)') ##this was found in paper [25] :test is between 1e-1 to 1e-3 


# fgsm_eps = 0.05
adversarial_training_enabled = False
run_log_dir = os.path.join(FLAGS.log_dir,
                           ('exp_bs_{bs}_lr_{lr}_' + ('adv_trained' if adversarial_training_enabled else ''))
                           .format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)


#shape = [filter_height, filter_width, in_channels, out_channels]
def weight_initialization(shape, w= 0.0625) :#because range is [-0.05; 0.05] from the paper
    initial = tf.random_normal(shape, mean= 0.0, stddev=w)
    return tf.Variable(initial)


def deepnn(x_image, class_count=43):
    padded_input = tf.pad(x_image, [[0, 0],[2, 2], [2, 2], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv2d(
        inputs=padded_input,
        filters=32,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        use_bias=False,
        name='conv1'
    )

    #conv1_weights = weight_initialization([5, 5, 32, 32])
    conv1_bn = tf.nn.relu(tf.layers.batch_normalization(conv1, name='conv1_bn'))
    pool1 = tf.layers.average_pooling2d(
        inputs=tf.pad(conv1_bn, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
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
        name='conv2'
    )
    #conv2_weights = weight_initialization([5, 5, 32, 32])
    conv2_bn = tf.nn.relu(tf.layers.batch_normalization(conv2, name='conv2_bn'))
    pool2 = tf.layers.average_pooling2d(
        inputs=tf.pad(conv2_bn, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
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
        name='conv3'
    )

    #conv3_weights = weight_initialization([5, 5, 64, 64])
    conv3_bn = tf.nn.relu(tf.layers.batch_normalization(conv3, name='conv3_bn'))
    pool3 = tf.layers.max_pooling2d(
        inputs=tf.pad(conv3_bn, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT"),
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
        name='conv4'
    )

    #conv4_weights = weight_initialization([4, 4, 64, 64])
    conv4_bn = tf.nn.relu(tf.layers.batch_normalization(conv4, name='conv4_bn'))
    
    logits = tf.layers.dense(inputs=conv4_bn, units=class_count, name='fc1')
    logits = tf.reshape(logits, [-1,class_count], name='logits')
    return logits


##preprocessing the data channel by channel
def preprocess(images, channels=3):
    print('preprocessing')
    print(images.shape())
    for i in range(0,3) :
        mean_channel = mean(images[:, :, i])
        stddev_channel = std(images[:, :, i])
        images[:, :, i] = (images[:, :, i]  - mean_channel) / stddev_channel
    return images


def main(_):
    tf.reset_default_graph()

    data = np.load('gtsrb_dataset.npz')
    ##data = preprocess(data) -->  not sure how i am going to be doing that

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
        x_image= tf.reshape(x, [-1, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 43])

    with tf.variable_scope('model'):
        logits = deepnn(x_image)
        print("logits:{}".format(logits.shape))
        model = CallableModelWrapper(deepnn, 'logits')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        decay_steps = 1000  # decay the learning rate every 1000 steps
        decay_rate = 0.9  # the base of our exponential for the decay
        global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
        
        ##variables that are defined in the paper -- not sure of the shape that the weights should be taking
        weights = tf.Variable(tf.random_normal([32, 32], mean = 0.0, stddev = 0.05), name='weights')

        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                           decay_steps, decay_rate, staircase=True)
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)
    # test_img_summary = tf.summary.image('Test Images', x_image)

    train_summary = tf.summary.merge([accuracy_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        #declare writers
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        #initialize the variables
        sess.run(tf.global_variables_initializer())
        step = 0
        
        for (train_images, train_labels) in batch_generator(data, 'train'):
            print("train:{}".format(step))

            step = step + 1
            _, train_summary_str = sess.run([train_step, train_summary], feed_dict={x_image: train_images, y_: train_labels})

        step = 0
        for (test_images, test_labels) in batch_generator(data, 'test'):
            if step % FLAGS.log_frequency == 0:
                train_writer.add_summary(train_summary_str, step)
                validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                       feed_dict={x_image: test_images, y_: test_labels})
                print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                validation_writer.add_summary(validation_summary_str, step)
            # Save the model checkpoint periodically.
            if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, checkpoint_path, global_step=step)

            if step % FLAGS.flush_frequency == 0:
                train_writer.flush()
                validation_writer.flush()
            step = step + 1
       
    #     # Resetting the internal batch indexes]

    #     cifar.reset()
    #     evaluated_images = 0
    #     test_accuracy = 0
    #     adversarial_test_accuracy = 0
    #     batch_count = 0

    #     while evaluated_images != cifar.nTestSamples:
    #         # Don't loop back when we reach the end of the test set
    #         (test_images, test_labels) = cifar.getTestBatch(allowSmallerBatches=True)
    #         test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})
    #         test_accuracy += test_accuracy_temp
            

    #         #adv_images = sess.run(x_adv, feed_dict={x: test_images, y_: test_labels})
    #         adv_images = sess.run(x_adv, feed_dict={x: test_images})
    #         x_adv_np = np.reshape(adv_images, [-1, cifar.IMG_WIDTH*cifar.IMG_HEIGHT*cifar.IMG_CHANNELS])

    #         adv_accuracy_temp = sess.run(accuracy, feed_dict={x: x_adv_np, y_: test_labels})
    #         adversarial_test_accuracy += adv_accuracy_temp

    #         adversarial_summary_str = sess.run(adversarial_summary, feed_dict={x: x_adv_np})
    #         adversarial_writer.add_summary(adversarial_summary_str, evaluated_images)

    #         test_summay_str = sess.run(img_summary, feed_dict={x: test_images})
    #         test_writer.add_summary(test_summay_str, evaluated_images)

    #         batch_count += 1
    #         evaluated_images += test_labels.shape[0]

    #     test_accuracy = test_accuracy / batch_count
    #     adversarial_test_accuracy = adversarial_test_accuracy / batch_count
    #     print('test set: accuracy on test set: %0.3f' % test_accuracy)
    #     print('adversarial test set: accuracy on adversarial test set: %0.3f' % adversarial_test_accuracy)
    #     print('model saved to ' + checkpoint_path)

    #     train_writer.close()
    #     validation_writer.close()
    #     adversarial_writer.close()
    #     test_writer.close()


if __name__ == '__main__':
    tf.app.run(main=main)

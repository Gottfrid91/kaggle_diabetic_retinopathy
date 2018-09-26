from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import argparse

import numpy as np
from six.moves import xrange
import tensorflow as tf
import sys
import resnet
import input
import train_ops

sys.dont_write_bytecode = True

##PARAMETERS TO BE SET
# training params
params = {}
#data path and data file names, needs to be set by each new user
params["DATA_DIR"] = "/home/olle/PycharmProjects/kaggle_drd_tfoffic/data"
params["TRAIN_FILE_NAME"] = '_train_10.tfrecords'
params["VALIDATION_FILE_NAME"] = '_validation_10.tfrecords'
# Fill in the entries one by one
params["task_name"] = "diabetic_retinopathy"
params["logging_dir"] = "./output" # generic output name, logging dir will be a combination of param entries
params["MAX_STEPS"] = 200000
params["BATCH_SIZE"] = 1
params["PRINTING_FREQ"] = 1000
params["SAVING_FREQ"] = 1000

#for input file functions, training task specific
params["IMAGE_SIZE"] = 256
params["NUM_CLASSES"] = 5
params["NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN"] = 35000
params["NUM_EXAMPLES_PER_EPOCH_FOR_EVAL"] = 3500
params["CAPACITY"] = 200

#for resnet file functions
params["MOVING_AVERAGE_DECAY"] = 0.9999
params["NUM_EPOCHS_PER_DECAY"] = 350.0
params["LEARNING_RATE_DECAY_FACTOR"] = 0.1
params["INITIAL_LEARNING_RATE"] = 0.1
params["_BATCH_NORM_DECAY"] = 0.997
params["_BATCH_NORM_EPSILON"] = 1e-5
params["MOMENTUM"] = 0.9
params["NUM_FILTERS"] = 64

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels for CIFAR-10.
        images, labels = input.distorted_inputs(params)
        val_images, val_labels = input.inputs(params)

        logits = resnet.inference(images,params,training=True)
        val_logits = resnet.inference(val_images,params,training=False)
        # calculate predictions
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        val_predictions = tf.cast(tf.argmax(val_logits, axis=1), tf.int32)

        # ops for batch accuracy calcultion
        correct_prediction = tf.equal(predictions, labels)
        val_correct_prediction = tf.equal(val_predictions, val_labels)

        batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        val_batch_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))

        # calculate training accuracy
        # Calculate loss.
        loss = train_ops.loss(logits, labels,params)

        val_loss = train_ops.loss(val_logits, val_labels,params)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train_ops.train(loss, global_step,params)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('val_loss', val_loss)
        tf.summary.scalar("training_accuracy", batch_accuracy)
        tf.summary.scalar('validation_accuracy', val_batch_accuracy)
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        # # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        sess = tf.Session()
        # sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(params["logging_dir"], sess.graph)

        step_start = 0
        try:
            ####Trying to find last checkpoint file fore full final model exist###
            print("Trying to restore last checkpoint ...")
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=params["logging_dir"])
            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)
            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            # get the step integer from restored path to start step from there
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)

            # create init op for the still unitilized variables
            init_new_vars_op = tf.variables_initializer(uninitialized_vars)
            sess.run(init_new_vars_op)
        except:
            # If all the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore any checkpoints. Initializing variables instead.")
            sess.run(init)

        #set lists for peroformance tracking
        accuracy_dev = []
        val_accuracy_dev = []
        for step in xrange(step_start, params["MAX_STEPS"]):
            start_time = time.time()
            _, loss_value,val_loss_value, accuracy, val_accuracy = sess.run([train_op, loss,val_loss,batch_accuracy,val_batch_accuracy])

            # append the next accuray to the development list
            val_accuracy_dev.append(val_accuracy)
            accuracy_dev.append(accuracy)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % params["PRINTING_FREQ"] == 0:
                num_examples_per_step = params["BATCH_SIZE"]
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f,val_loss = %.2f, avg_batch_accuracy = %.2f, avg_batch_accuracy_validation = %.2f'
                              ',(%.1f examples/sec; %.3f '
                              'sec/batch)')
                # take averages of all the accuracies from the previous bathces
                print(format_str % (datetime.now(), step, loss_value, val_loss_value,
                                    np.mean(accuracy_dev),np.mean(val_accuracy_dev),
                                    examples_per_sec, sec_per_batch))
            if step % params["PRINTING_FREQ"] == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % params["SAVING_FREQ"] == 0 or (step + 1) == params["MAX_STEPS"]:

                directory = os.path.join(params["logging_dir"],params["task_name"],
                                               str(params["INITIAL_LEARNING_RATE"]))
                # set paths and saving ops for the full and sub_network
                checkpoint_path = os.path.join(directory,'model.ckpt')

                #if output dir does not exit, create it
                try:
                    os.stat(directory)
                except:
                    os.makedirs(directory)

                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()

if __name__ == '__main__':
    tf.app.run()

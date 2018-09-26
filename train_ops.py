# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def loss(logits, labels,params):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # We use a weight decay of 0.0001
  weight_decay = 0.0001

  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=labels, logits=logits)
  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  def loss_filter_fn(_):
      return True
  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)]) ######## CHECK SO IDENTIVCAL
  tf.summary.scalar('l2_loss', l2_loss)

  loss = cross_entropy + l2_loss
  return loss



def train(loss, global_step,params):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  ##IMPLEMENT EXACT TF OFFIC TRAINING OPS AND TRAIN, KILL ALL OUNECESSARY CODE
  boundary_epochs = [30, 60, 80,100]
  decay_rates = [1.0, 0.1, 0.01, 0.001,0.0001]

  # Variables that affect learning rate.
  num_batches_per_epoch = params["NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN"] / params["BATCH_SIZE"]

  boundaries = [int(num_batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [params["INITIAL_LEARNING_RATE"] * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  learning_rate = learning_rate_fn(global_step)

  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=params["MOMENTUM"]
  )
  # Apply gradients.

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  minimize_op = optimizer.minimize(loss, global_step)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  train_op = tf.group(minimize_op, update_ops)

  return train_op

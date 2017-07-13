import os
import re
import tensorflow as tf


def initialize_from_checkpoint(sess,
                               checkpoint,
                               checkpoint_scope='',
                               model_scope='',
                               include_patterns=None,
                               exclude_patterns=None):
    """
    Initialize relevant variables of a model from a checkpoint.

    Restores common variables between the model and a checkpoint.
    Limited support for translating variable names in the checkpoint,
     to variable names in current model.

    :param sess:
        tf.Session object
    :param checkpoint:
        path to the checkpoint file or directory
    :param checkpoint_scope:
        prefix of checkpoint variable names to translate
    :param model_scope:
        corresponding prefix of variables names in the model
    :param include_patterns:
        variables to include in the restore
    :param exclude_patterns:
        variables to exclude in the restore
    """
    if os.path.isdir(checkpoint):
        checkpoint = tf.train.latest_checkpoint(checkpoint)
    reader = tf.train.NewCheckpointReader(checkpoint)
    foreign_var_names = reader.get_variable_to_shape_map().keys()

    '''
    If checkpoint variables have another scope,
    translate them to this network's scope. '''
    translation = {var.replace(checkpoint_scope, model_scope): var
                   for var in foreign_var_names}

    all_domestic_vars = tf.contrib.framework.get_variables()
    domestic_vars = tf.contrib.framework.filter_variables(
        var_list=all_domestic_vars,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    # Remove the unique numerical identifier - they aren't in checkpoint names
    domestic_var_names = [re.sub(':\d+', '', var.name)
                          for var in domestic_vars]

    variables_to_restore = {
        translation[domestic_var_name]:
            tf.contrib.framework.get_unique_variable(domestic_var_name)
        for domestic_var_name in domestic_var_names
        if domestic_var_name in translation
    }
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, checkpoint)


def accuracy(logits, labels):
    """
    Calculate rank-1.

    Logits and labels must be same dtype.

    :param logits: Tensor of class probabilities
    :param labels: Tensor of class labels
    :return: rank-1
    """
    with tf.name_scope('accuracy'):
        predictions = tf.argmax(logits, 1, name='predictions')
        is_correct = tf.to_float(tf.equal(predictions, labels), name='is_correct')
        num_correct = tf.reduce_sum(is_correct, name='num_correct')
        num_examples = tf.to_float(tf.size(labels), name='num_examples')
        acc = tf.divide(num_correct, num_examples, name='accuracy')
        return acc


def l1(group):
    return tf.sqrt(tf.reduce_sum(tf.square(group)), name='l1')

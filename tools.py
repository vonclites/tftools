import os
import re
import tensorflow as tf


def get_warm_start_mapping(checkpoint_file,
                           checkpoint_scope='',
                           variable_scope='',
                           include_patterns=None,
                           exclude_patterns=None):
    """
    Returns dict mapping the names of variables in the checkpoint
    to the corresponding names of variables in the model being
    warm-started.

    Can be used as assignment_map parameter in tf.train.init_from_checkpoint

    :param checkpoint_file:
        path to the checkpoint file or directory
    :param checkpoint_scope:
        prefix of checkpoint variable names to translate
    :param variable_scope:
        corresponding prefix of variables names in the model
    :param include_patterns:
        variables to include in the restore
    :param exclude_patterns:
        variables to exclude in the restore
    """
    if os.path.isdir(checkpoint_file):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_file)
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    foreign_var_names = reader.get_variable_to_shape_map().keys()

    '''
    If checkpoint variables have another scope,
    translate them to this network's scope. '''
    translation = {var.replace(checkpoint_scope, variable_scope): var
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
    return variables_to_restore


def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

      Calculates a confusion matrix. For estimation over a stream of data,
      the function creates an  `update_op` operation.

      Args:
        labels: A `Tensor` of ground truth labels with shape [batch size] and of
          type `int32` or `int64`. The tensor will be flattened if its rank > 1.
        predictions: A `Tensor` of prediction results for semantic labels, whose
          shape is [batch size] and type `int32` or `int64`. The tensor will be
          flattened if its rank > 1.
        num_classes: The possible number of labels the prediction task can
          have. This value must be provided, since a confusion matrix of
          dimension = [num_classes, num_classes] will be allocated.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
          `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
          be either `1`, or the same as the corresponding `labels` dimension).

      Returns:
        total_cm: A `Tensor` representing the confusion matrix.
        update_op: An operation that increments the confusion matrix.
      """
    # Local variable to accumulate the predictions in the confusion matrix.
    cm_dtype = tf.int64 if weights is not None else tf.float64
    total_cm = _create_local(
        'total_confusion_matrix',
        shape=[num_classes, num_classes],
        dtype=cm_dtype)

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = tf.to_int64(predictions)
    labels = tf.to_int64(labels)
    num_classes = tf.to_int64(num_classes)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = tf.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = tf.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = tf.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = tf.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=cm_dtype)
    update_op = tf.assign_add(total_cm, current_cm)
    return total_cm, update_op


def per_class_accuracies(labels,
                         predictions,
                         num_classes,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
    """Calculates per-class accuracies.

      Calculates the accuracy for each class.

      For estimation of the metric over a stream of data, the function creates an
      `update_op` operation that updates these variables and returns the
      `accuracies`.

      If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

      Args:
        labels: A `Tensor` of ground truth labels with shape [batch size] and of
          type `int32` or `int64`. The tensor will be flattened if its rank > 1.
        predictions: A `Tensor` of prediction results for semantic labels, whose
          shape is [batch size] and type `int32` or `int64`. The tensor will be
          flattened if its rank > 1.
        num_classes: The possible number of labels the prediction task can
          have. This value must be provided, since a confusion matrix of
          dimension = [num_classes, num_classes] will be allocated.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
          `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
          be either `1`, or the same as the corresponding `labels` dimension).
        metrics_collections: An optional list of collections that
          `accuracies' should be added to.
        updates_collections: An optional list of collections `update_op` should be
          added to.
        name: An optional variable_scope name.

      Returns:
        accuracies: A `Tensor` representing the per class accuracies.
        update_op: An operation that increments the confusion matrix.

      Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, or if
          `weights` is not `None` and its shape doesn't match `predictions`, or if
          either `metrics_collections` or `updates_collections` are not a list or
          tuple.
      """
    with tf.variable_scope(name, 'class_accuracy',
                           (predictions, labels, weights)):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = streaming_confusion_matrix(
            labels, predictions, num_classes, weights=weights)

        per_row_sum = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = per_row_sum

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0), denominator,
            tf.ones_like(denominator))
        accuracies = tf.div(cm_diag, denominator)

        if metrics_collections:
            tf.get_default_graph().add_to_collections(
                metrics_collections, accuracies)

        if updates_collections:
            tf.get_default_graph().add_to_collections(
                updates_collections, update_op)

        return accuracies, update_op


def accuracy(logits, labels):
    """
    Calculate rank-1 accuracy.

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


def group_sparsity(groups, cost, collections=None):
    # TODO: Add docs
    collections = list(collections or [])
    collections += [tf.GraphKeys.LOSSES]
    group_losses = dict()

    for group_name, group in groups.items():
        group_loss = tf.reduce_sum(
            input_tensor=tf.norm(group, ord='euclidean', axis=0)
        ) / tf.to_float(tf.shape(group)[0])
        group_loss = tf.multiply(cost, group_loss,
                                 name='{}_sparsity_loss'.format(group_name))
        group_losses[group_name] = group_loss

    total_loss = tf.reduce_sum(list(group_losses.values()),
                               name='group_sparsity_loss')

    for collection in collections:
        tf.add_to_collection(collection, total_loss)

    return total_loss, group_losses


def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=tf.float32):
    """Creates a new local variable.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      collections: A list of collection names to which the Variable will be added.
      validate_shape: Whether to validate the shape of the variable.
      dtype: Data type of the variables.

    Returns:
      The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])

    var = tf.get_local_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=tf.zeros_initializer,
        validate_shape=validate_shape)

    for collection in collections:
        tf.add_to_collection(collection, var)

    return var

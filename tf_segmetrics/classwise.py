import tensorflow as tf


def batched_classwise(y_pred, y_true, parts, function):
    size = y_pred.shape[0]
    
    results = None
    first = True
    
    part_size = int(size/parts)
    
    for lower, upper in zip(range(0, size, part_size), range(part_size, size + part_size, part_size)):
        if first:
            results = function(y_pred[lower:upper], y_true[lower:upper])
            first = False
            continue
            
        results += function(y_pred[lower:upper], y_true[lower:upper])
    
    return results / parts

def classwise_dice(y_pred, y_true, smooth=1e-5):
    classes = y_pred.shape[-1]

    y_true = tf.one_hot(y_true[..., -1], classes)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    intersection = 2 * tf.cast(intersection, tf.float32) + smooth
    
    denominator = tf.cast(
        tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2]),
        tf.float32
    ) + smooth

    dice = intersection / denominator
    
    return dice

def classswise_precision(y_pred, y_true):
    classes = y_pred.shape[-1]

    y_true = tf.one_hot(y_true[..., -1], classes)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    true_positives = tf.reduce_sum(y_true * y_pred, [0, 1, 2])
    predicted_true = tf.reduce_sum(y_pred, [0, 1, 2])
    
    return true_positives / predicted_true

def classwise_recall(y_pred, y_true):
    classes = y_pred.shape[-1]

    y_true = tf.one_hot(y_true[..., -1], classes)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    true_positives = tf.reduce_sum(y_true * y_pred, [0, 1, 2])
    existing_positives = tf.reduce_sum(y_true, [0, 1, 2])
    
    return true_positives / existing_positives
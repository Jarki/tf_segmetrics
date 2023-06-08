import tensorflow as tf
import numpy as np


def dice(y_true, y_pred, smooth=1e-5):
  classes = y_pred.shape[-1]

  y_true = tf.cast(y_true, tf.int32)[..., -1]
  y_true = tf.one_hot(y_true, classes)

  y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
  y_pred = tf.cast(y_pred > 0.5, tf.float32)

  intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 0])
  intersection = 2 * tf.cast(intersection, tf.float32) + smooth

  denominator = tf.cast(
    tf.reduce_sum(y_true, axis=[1, 2, 0]) + tf.reduce_sum(y_pred, axis=[1, 2, 0]),
    tf.float32
  ) + smooth

  dice = intersection / denominator

  return tf.reduce_mean(dice, axis=-1)

def iou(y_true, y_pred):
    classes = y_pred.shape[-1]

    y_true = tf.cast(y_true, tf.int32)[..., -1]
    y_true = tf.one_hot(y_true, classes)
    
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 0])
    
    union = tf.reduce_sum(y_true, axis=[1, 2, 0]) + tf.reduce_sum(y_pred, axis=[1, 2, 0]) - intersection 
    
    iou = intersection / union
    
    return tf.experimental.numpy.nanmean(iou, axis=-1)
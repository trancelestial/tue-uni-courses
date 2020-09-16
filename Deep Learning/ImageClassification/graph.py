import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

import time

INPUT_SHAPE = (4,)
OUTPUT = 3

def create_model(c_out, input_shape):
    c1 = 4
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(c1),
        layers.Activation('relu'),
        layers.Dense(c_out),
        layers.Activation('relu')
    ])
    return model

def loss(y, y_hat):
    return tf.tensordot(y_hat - y, y_hat - y, axes=0) / 2

model = create_model(OUTPUT, INPUT_SHAPE)

logdir = "./tb/%d/" % time.time()
writer = tf.summary.create_file_writer(logdir)  # Needed for Tensorboard logging

@tf.function
def graph_trace_function(x, y):
    with tf.GradientTape() as tp:
        logits = model(x, training=True)
        loss_value = tf.keras.losses.MSE(y, logits)
        tp.gradient(loss_value, trainable)
    return loss_value

inp = tf.constant([[0, 0, 0, 0]], dtype=tf.float32, name='input')
y = tf.constant([[0, 0, 0]], dtype=tf.float32, name='y')

tf.summary.trace_on(graph=True, profiler=True)
trainable = model.trainable_weights
z = graph_trace_function(inp, y)
# grads = tf.gradients(z, trainable)
with writer.as_default():
    tf.summary.trace_export(name='network_trace', step=0, profiler_outdir=logdir)

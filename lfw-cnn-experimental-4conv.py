
# In[23]:


from __future__ import division
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import sys
import numpy as np
import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt
from scipy.misc import toimage

LEARNING_RATE = .001
BATCH_SIZE = 128

# color = true keeps color channels seperate
# with min_faces set to 14 - there will be 106 possible classifications
# image dimensions with param resize=0.5 will give 64x64 images
people = fetch_lfw_people(
    color=True,
    resize=0.5,
    slice_=(slice(61,189), slice(61,189)),
    min_faces_per_person=14
)
# import _pickle as pickle
#
# with open('people.pickle', 'rb') as f:
#     people = pickle.load(f, encoding='iso-8859-1')

X = people.images
y = np.asarray(people.target, dtype=np.int32)

labels = people.target_names


# To visualize, here's the dataset we're playing with

# In[4]:


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.figure(figsize=(18, 18))

for i in range(0,121):
    d = plt.subplot(11, 11, i + 1)
    d.set_xticks([])
    d.set_yticks([])
    plt.imshow(toimage(X[i]))

plt.tight_layout()
plt.show()


# In[37]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from tensorflow.contrib import learn


def model(features, labels, mode):
    # [64, 64, 3]
    c1 = tf.layers.conv2d(
        inputs=features['x'],
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    print('c1', c1.get_shape())

    c2 = tf.layers.conv2d(
        inputs=c1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    print('c2', c2.get_shape())

    p1 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=2) # strides will reduce size by n / k
    # p1 Output Tensor Shape: [batch_size, 32, 32, 32]
    print('p1', p1.get_shape())

    d1 = tf.layers.dropout(inputs=p1, rate=.3, training = mode == learn.ModeKeys.TRAIN)
    print('d1', d1.get_shape())

    c3 = tf.layers.conv2d(
        inputs=d1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )
    print('c3', c3.get_shape())
    c4 = tf.layers.conv2d(
      inputs=c3,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu
    )
    print('c4', c4.get_shape())

    p2 = tf.layers.max_pooling2d(inputs=c4, pool_size=[2, 2], strides=2)
    print('p2', p2.get_shape())

    d2 = tf.layers.dropout(
      inputs=p2, rate=.3, training=mode == learn.ModeKeys.TRAIN)
    print('d2', d2.get_shape())

    flat = tf.reshape(d2, [-1, 16 * 16 * 64])
    print('flat', flat.get_shape())

    dense_layer = tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )
    print('dense_layer', dense_layer.get_shape())

    d3 = tf.layers.dropout(inputs=dense_layer, rate=.4, training = mode == learn.ModeKeys.TRAIN)
    print('d3', d3.get_shape())

    dense_l2 = tf.layers.dense(inputs=d3, units=512, activation=tf.nn.relu)
    print('dense_l2', dense_l2.get_shape())

    d4 = tf.layers.dropout(inputs=dense_l2, rate=.6, training = mode == learn.ModeKeys.TRAIN)
    print('d4', d4.get_shape())

    logits = tf.layers.dense(inputs=d4, units=106)
    print('logits', logits.get_shape())

    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=106)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=logits
    )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels,
          predictions=predictions["classes"]
      )
    }

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# We define the model as such

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

classifier_lfw = tf.estimator.Estimator(
    model_fn=model,
    model_dir="cnn-experimental-4conv"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_train},
    y=y_train,
    batch_size=BATCH_SIZE,
    num_epochs=None,
    shuffle=True
)
print('Starting training...')
classifier_lfw.train(
    input_fn=train_input_fn,
    steps=6000,
    hooks=[
        tf.train.LoggingTensorHook(
            tensors = {
                'probabilities': 'softmax_tensor'
            },
            every_n_iter=50
        )
    ]
)


# Using Tensorflow's input estimator, we can guage our accuracy

# In[ ]:


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_test},
    y=y_test
    num_epochs=1,
    shuffle=False
)

print(classifier_lfw.evaluate(
    input_fn=eval_input_fn,
))

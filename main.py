#!/usr/bin/env python

import os
import argparse
import build_tf_dataset
from models import vgg16
import tensorflow as tf


def train():
    train_dataset = build_tf_dataset.tf_dataset()

    IMG_SHAPE = (244, 244, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(include_top=True,
                                              weights='imagenet')
    prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
    model = tf.keras.Sequential([
        VGG16_MODEL,
        prediction_layer
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    history = model.fit(train_dataset,
                        epochs=100,
                        steps_per_epoch=50)

    # train_iter = iter(train_dataset)
    steps = 1000
    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # vgg = vgg16.vgg16(imgs, 'vgg16_weights.npz')
    for i in range(steps):
        print("Step {}".format(i))
        images, labels = next(iter(train_dataset))
        print(images.numpy().shape, labels.numpy().shape)
        # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]


if __name__ == "__main__":
    train()

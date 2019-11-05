#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


tf.executing_eagerly()

BATCH_SIZE = 32


def get_label(file_path):
    print(file_path)
    # convert the path to a list of path components
    class_label = tf.strings.split(file_path, '_')
    #  Image_0_1_8.png
    # The second to last is the class-directory
    print("label")
    print(class_label)
    # class_label = tf.constant(["0"])
    # return class_label[1]
    return class_label[1] == np.array(["0", "1"], dtype='<U10')

def decode_img(img):
    print("Image")
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [224, 224])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def tf_dataset():
    list_ds = tf.data.Dataset.list_files("images/*.png")
    for f in list_ds.take(5):
        print(f.numpy())
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(5):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    train_ds = prepare_for_training(labeled_ds)

    image_batch, label_batch = next(iter(train_ds))

    print(image_batch.numpy().shape, label_batch.numpy().shape)

    image_batch, label_batch = next(iter(train_ds))

    print(image_batch.numpy().shape, label_batch.numpy().shape)

    return train_ds


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if __name__ == "__main__":
    tf_dataset()

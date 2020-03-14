import tensorflow as tf
import numpy as np
import os
from model.mobilefacenet import *
from model.mobilefacenet_func import *
from sklearn.model_selection import train_test_split

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

cls_num = 10572

def mobilefacenet_train(resume=False):
    x = inputs = tf.keras.layers.Input(shape=(112, 96, 3))
    x = mobilefacenet(x)

    if not resume:
        x = tf.keras.layers.Dense(cls_num)(x)
        outputs = tf.nn.softmax(x)
        return tf.keras.models.Model(inputs, outputs)
    else:
        y = tf.keras.layers.Input(shape=(cls_num,))
        outputs = ArcFace(n_classes=cls_num)((x, y))

        return tf.keras.models.Model([inputs, y], outputs)

if __name__ == '__main__':

    model = keras.models.load_model("pretrained_model/train_1/best_model_.50-8.39.h5", custom_objects={"ArcFace_v2": ArcFace_v2})
    #model.load_weights("pretrained_model/")
    for layer in model.layers:
        print(layer)
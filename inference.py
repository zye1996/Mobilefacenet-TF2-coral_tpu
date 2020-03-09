import tensorflow as tf
import numpy as np
import os
from model.mobilefacenet import *
from sklearn.model_selection import train_test_split

cls_num = 10572

# construct model
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
    
    model = mobilefacenet_train()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001, epsilon = 1e-8), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    init_tensor = [tf.random.normal(shape=(1, 112, 96, 3)), tf.zeros(shape=(1, cls_num, ))]
    model(init_tensor)
    model.load_weights('pretrained_model/model_ckpt.h5')
    model.save('pretrained_model/whole_model.h5')
    
import tensorflow as tf
import numpy as np
import os
from model.mobilefacenet import *
from sklearn.model_selection import train_test_split

# CONFIG
RESUME = True

# load dataset
data_root = "/Users/zhenyiye/Downloads/Mac(1)/CASIA_SUB"
img_txt_dir = os.path.join(data_root, 'CASIA-WebFace-112X96.txt')


def load_dataset(val_split=0.05):
    image_list = []     # image directory
    label_list = []     # label
    with open(img_txt_dir) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        image_dir, label_name = info.split(' ')
        image_list.append(os.path.join(data_root, 'CASIA-WebFace-112X96', image_dir))
        label_list.append(int(label_name))

    trainX, testX, trainy, testy = train_test_split(image_list, label_list, test_size=val_split)

    return trainX, testX, trainy, testy


def preprocess(x,y):
    # x: directory，y：label
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [112, 96])

    x = tf.image.random_flip_left_right(x)

    # x: [0,255]=> -1~1
    x = (tf.cast(x, dtype=tf.float32) - 127.5) / 128.0
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=cls_num)

    if RESUME:
        return (x, y), y
    else:
        return x, y

# get data slices
train_image, val_image, train_label, val_lable = load_dataset()

# get class number
cls_num = len(np.unique(train_label))

batchsz = 64
db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))     # construct train dataset
db_train = db_train.shuffle(3).map(preprocess).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
db_val = db_val.shuffle(3).map(preprocess).batch(batchsz)


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

    model = mobilefacenet_train(resume=True)
    print(model.summary())

    # callbacks
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))


    history = LossHistory()
    callback_list = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10),
                     tf.keras.callbacks.ModelCheckpoint("pretrained_model/model_ckpt.h5", monitor='val_accuracy', save_best_only=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 200, min_lr = 0),
                     LossHistory()]

    # compile model
    optimizer = tf.keras.optimizers.Adam(lr = 0.001, epsilon = 1e-8)
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100, callbacks=callback_list)



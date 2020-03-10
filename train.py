import tensorflow as tf
import numpy as np
import os
from model.mobilefacenet import *
from model.mobilefacenet_func import *
from sklearn.model_selection import train_test_split
from test_lfw import *

# CONFIG
RESUME = True

# load dataset
data_root = "C:/Users/chubb/PycharmProjects/mbfacenet_tf2/CASIA"
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

batchsz = 128
db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))     # construct train dataset
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
db_val = db_val.shuffle(1000).map(preprocess).batch(batchsz)


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

    class SaveModel(keras.callbacks.Callback):

        def on_epoch_end(self, batch, logs=None):
            model.save_weights("pretrained_model/", save_format="tf")

    # test on LWF
    class TestLWF(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs=None):
            infer_model = tf.keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)
            get_features(infer_model, "C:/Users/chubb/PycharmProjects/mbfacenet_tf2/lfw", 'result/best_result.mat')
            evaluation_10_fold()


    history = LossHistory()
    callback_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
                     tf.keras.callbacks.ModelCheckpoint("pretrained_model/best_model_.{epoch:02d}-{val_loss:.2f}.h5",
                                                        monitor='val_loss', save_best_only=True), #, save_weights_only=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.2, patience=200, min_lr=0),
                     LossHistory(),
                     TestLWF()]

    # compile model
    # optimizer = tf.keras.optimizers.Adam(lr = 0.001, epsilon = 1e-8)
    optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=4e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100, callbacks=callback_list)

    # inference model save
    inference_model = keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)
    inference_model.save('pretrained_model/inference_model.h5')




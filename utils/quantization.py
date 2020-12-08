import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

root_dir = "../image"

file_name = '/home/yzy/Downloads/240_320_retinaface_k210_exp/infer_model.h5'

def data_generator():
    for file in os.listdir(root_dir):
        img = plt.imread(os.path.join(root_dir, file), format='rgb').astype('float32')
        img = (img.astype('float32') - 127.5) / 128.0
        yield [img[np.newaxis, :, :, :]]


if __name__ == "__main__":

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(file_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = data_generator
    converter.experimental_new_converter = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    # write to file
    with open(file_name.split('.')[0]+'_quant.tflite', "wb") as f:
        f.write(tflite_quant_model)

import tensorflow as tf

from model.mobilefacenet import *

model_path = "train_3_arcface/best_model_.58-6.94.h5"
output_model_path = "training_model/inference_model_0.993.h5"

if __name__ == '__main__':
    model = tf.keras.models.load_model(model_path, custom_objects={"ArcFace_v2": ArcFace_v2})
    inference_model = tf.keras.models.Model(inputs=model.input[0], outputs=model.output)
    inference_model.save(output_model_path)

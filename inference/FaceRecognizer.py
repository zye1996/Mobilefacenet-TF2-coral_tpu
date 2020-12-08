import platform

import tensorflow as tf
import tflite_runtime.interpreter as tflite

from postprocessing import *

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


def get_quant_int8_output(interpreter, output_index):
    feature = interpreter.get_tensor(output_index)
    if feature.dtype == np.uint8:
        zero_points = interpreter.get_output_details()[0]["quantization_parameters"]["zero_points"]
        scales = interpreter.get_output_details()[0]["quantization_parameters"]["scales"]
        return (feature - zero_points) * scales
    return feature

class FaceDetector():

    def __init__(self, model_dir, image_size=None, tpu=True):

        if image_size is None:
            image_size = [320, 240]
        if tpu:
            self.interpreter = tflite.Interpreter(model_path=model_dir,
                                             experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
        else:
            self.interpreter = tf.compat.v1.lite.Interpreter(model_path=model_dir)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.bbox_index = self.interpreter.get_output_details()[0]['index']
        self.ldmk_index = self.interpreter.get_output_details()[1]['index']
        self.prob_index = self.interpreter.get_output_details()[2]['index']
        self.image_size = image_size

    def detect_face(self, image):

        # feed forwardR
        self.interpreter.set_tensor(self.input_index, image[np.newaxis, :, :, :])
        self.interpreter.invoke()

        # get result
        bbox = self.interpreter.get_tensor(self.bbox_index)
        ldmk = self.interpreter.get_tensor(self.ldmk_index)
        prob = self.interpreter.get_tensor(self.prob_index)

        # post processing
        pred_prob, pred_bbox, pred_ldmk = pred_boxes(bbox[0, ...], prob[0, ...], ldmk[0, ...])

        # calculate bbox corrdinate
        pred_bbox_pixel = pred_bbox * np.tile(self.image_size, 2)
        pred_ldmk_pixel = pred_ldmk * np.tile(self.image_size, 5)

        # nms
        keep = nms_oneclass(pred_bbox_pixel, pred_prob)
        if len(keep) > 0:
            pred_bbox_pixel = pred_bbox_pixel[keep, :]
            pred_ldmk_pixel = pred_ldmk_pixel[keep, :]
            pred_prob = pred_prob[keep]
        else:
            return [], [], []

        return pred_bbox_pixel, pred_ldmk_pixel, pred_prob


class FaceRecognizer:

    def __init__(self, model_dir, tpu=True, mask=False):

        self.tpu = tpu
        self.mask = mask
        if self.tpu:
            self.interpreter = tflite.Interpreter(model_path=model_dir,
                                                  experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
        else:
            self.interpreter = tf.compat.v1.lite.Interpreter(model_path=model_dir)
        self.interpreter.allocate_tensors()
        self.rec_input_index = self.interpreter.get_input_details()[0]['index']
        if self.mask:
            self.rec_output_index = self.interpreter.get_output_details()[1]['index']
            self.mask_output_index = self.interpreter.get_output_details()[0]['index']
            print('here')
        else:
            self.rec_output_index = self.interpreter.get_output_details()[0]['index']

    def face_recognize(self, image, landmark=None):

        if landmark is not None:
            aligned = face_algin_by_landmark(image, landmark)
        else:
            aligned = image

        if not self.tpu:
            aligned_norm = np.expand_dims(aligned, axis=0).astype(np.float32)
        else:
            aligned_norm = np.expand_dims(aligned, axis=0)

        self.interpreter.set_tensor(self.rec_input_index, aligned_norm)
        self.interpreter.invoke()
        feature = get_quant_int8_output(self.interpreter, self.rec_output_index)
        if self.mask:
            mask = get_quant_int8_output(self.interpreter, self.mask_output_index)
            return feature, mask
        return feature, None

import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import argparse
import platform
import json
import threading
import multiprocessing

from postprocessing import *
from FileVideoStreamer import *

# FLAGS
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--platform', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--w', type=int, default=320)
parser.add_argument('--coral_tpu', type=bool, default=False,
                    help="whether use tpu")
args = parser.parse_args()

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def get_quant_int8_output(interpreter, output_index):
    feature = interpreter.get_tensor(output_index)
    if feature.dtype == np.uint8:
        zero_points = interpreter.get_output_details()[0]["quantization_parameters"]["zero_points"]
        scales = interpreter.get_output_details()[0]["quantization_parameters"]["scales"]
        return (feature - zero_points) * scales
    return feature


def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


def cam_loop(q_flipped, q_rgbf):

    cap = cv2.VideoCapture(0)

    while True:
        _ , frame = cap.read()
        if frame is not None:
            if args.platform == 'picamera':
                rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # roate counterclockwise
                h, w, _ = rotated.shape
                cropped_h = w * H / W
                cropped = rotated[int((h - cropped_h) / 2):int((h + cropped_h) / 2), :]
            else:
                cropped = frame[:, 640 - 480:640 + 480]  # crop to retain the center area
            resized = cv2.resize(cropped, (W, H))  # resize the images
            flipped = cv2.flip(resized, 1)  # flip the camera
            rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB).astype('float32')
            rgb_f = (rgb / 255) - 0.5
            q_flipped.put(flipped)
            q_rgbf.put(rgb_f)


if __name__ == '__main__':

    MODEL_PATH_TPU = "pretrained_model/facedetection_320_240_edgetpu_cocompiled.tflite"
    REC_MODEL_PATH_TPU = "pretrained_model/mobilefacenet_edgetpu_cocompiled.tflite"
    MODEL_PATH = "pretrained_model/ulffd_landmark.tflite"
    REC_MODEL_PATH = "pretrained_model/inference_model_0_quant.tflite"
    DATABASE_PATH = "pretrained_model/db.npy"
    LABEL_PATH = "pretrained_model/label.json"

    if args.w == 320:
        W, H = 320, 240
    else:
        W, H = 640, 640

    # load model
    if args.coral_tpu:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH_TPU,
                                         experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
        interpreter_rec = tflite.Interpreter(model_path=REC_MODEL_PATH_TPU,
                                                         experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
    else:
        interpreter = tf.compat.v1.lite.Interpreter(model_path=MODEL_PATH)
        interpreter_rec = tf.compat.v1.lite.Interpreter(model_path=REC_MODEL_PATH)

    # load database
    rec_db = np.load(DATABASE_PATH)
    label = json.load(open(LABEL_PATH))

    # get handler
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    bbox_index = interpreter.get_output_details()[0]['index']
    ldmk_index = interpreter.get_output_details()[1]['index']
    prob_index = interpreter.get_output_details()[2]['index']
    interpreter_rec.allocate_tensors()
    rec_input_index = interpreter_rec.get_input_details()[0]['index']
    rec_output_index = interpreter_rec.get_output_details()[0]['index']

    # Quene
    q_flipped = multiprocessing.Manager().Queue(1)
    q_rgbf = multiprocessing.Manager().Queue(1)

    # video capture
    # cap = cv2.VideoCapture(0)
    # cap = FileVideoStream(0).start()
    cam_process = multiprocessing.Process(target=cam_loop,args=(q_flipped, q_rgbf, ))
    cam_process.start()

    time.sleep(1)

    while True:

        start = time.time()

        # read camera
        # ret, frame = cap.read()
        flipped = q_flipped.get()
        rgb_f = q_rgbf.get()

        
        # image processing
        '''
        if args.platform == 'picamera':
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # roate counterclockwise
            h, w, _ = rotated.shape
            cropped_h = w * H / W
            cropped = rotated[int((h - cropped_h) / 2):int((h + cropped_h) / 2), :]
        else:
            cropped = frame[:, 640 - 480:640 + 480]  # crop to retain the center area
        resized = cv2.resize(cropped, (W, H))  # resize the images
        flipped = cv2.flip(resized, 1)  # flip the camera
        '''

        #rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB).astype('float32') #flipped[...,::-1].copy().astype('float32') #
        #rgb_f = (rgb / 255) - 0.5  # normalization


        # feed forward
        interpreter.set_tensor(input_index, rgb_f[np.newaxis, :, :, :])
        interpreter.invoke()

        # get result
        bbox = interpreter.get_tensor(bbox_index)
        ldmk = interpreter.get_tensor(ldmk_index)
        prob = interpreter.get_tensor(prob_index)

        # post processing
        pred_prob, pred_bbox, pred_ldmk = pred_boxes(bbox[0, ...], prob[0, ...], ldmk[0, ...])

        # calculate bbox corrdinate
        pred_bbox_pixel = pred_bbox * np.tile([320, 240], 2)
        pred_ldmk_pixel = pred_ldmk * np.tile([320, 240], 5)

        # nms
        keep = nms_oneclass(pred_bbox_pixel, pred_prob)

        # detection
        if len(keep > 0):

            pred_bbox_pixel = pred_bbox_pixel[keep, :]
            pred_ldmk_pixel = pred_ldmk_pixel[keep, :]
            pred_prob = pred_prob[keep]

            # crop faces
            valid_index, vaild_bboxs, face_imgs, face_landmarks = crop_faces(flipped, pred_bbox_pixel, pred_ldmk_pixel)
            pred_bbox_pixel = pred_bbox_pixel[valid_index, :]
            pred_ldmk_pixel = pred_ldmk_pixel[valid_index, :]
            pred_prob = pred_prob[valid_index]

            # loop over faces
            for i in (range(pred_prob.shape[0])):

                # face recognition
                aligned = face_algin_by_landmark(face_imgs[i], face_landmarks[i])
                if not args.coral_tpu:
                    aligned_norm = preprocess(aligned)
                else:
                    aligned_norm = np.expand_dims(aligned, axis=0)
                interpreter_rec.set_tensor(rec_input_index, aligned_norm)
                interpreter_rec.invoke()
                # feature = rec_model.predict(aligned_norm)
                feature = get_quant_int8_output(interpreter_rec, rec_output_index)
                result = face_recognition(feature, rec_db)
                print(result)

                # put label
                cv2.putText(flipped, label[str(result[0])], (int((pred_bbox_pixel[i, 0])), int(pred_bbox_pixel[i, 1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                # put rectangle
                cv2.rectangle(flipped, tuple(pred_bbox_pixel[i, :2].astype(int)),
                              tuple(pred_bbox_pixel[i, 2:].astype(int)), (255, 0, 0), 2)
                for ldx, ldy, color in zip(pred_ldmk_pixel[i][0::2].astype(int),
                                           pred_ldmk_pixel[i][1::2].astype(int),
                                           [(255, 0, 0), (0, 255, 0),
                                            (0, 0, 255), (255, 255, 0),
                                            (255, 0, 255)]):
                    cv2.circle(flipped, (ldx, ldy), 3, color, 2)

        cv2.imshow('detection', flipped)

        # print fps
        print("fps:", 1 / (time.time() - start))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    #cap.release()
    cam_process.terminate()
    cam_process.join()
    cv2.destroyAllWindows()

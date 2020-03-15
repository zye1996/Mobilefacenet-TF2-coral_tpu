import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import argparse

from postprocessing import *

# FLAGS
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--platform', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--w', type=int, default=320)
parser.add_argument('--coral_tpu', type=bool, default=False,
                    help="whether use tpu")
args = parser.parse_args()

def get_quant_int8_output():
    pass

def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':

    MODEL_PATH = "pretrained_model/ulffd_landmark.tflite"
    REC_MODEL_PATH = "pretrained_model/inference_model_quant.tflite"
    DATABASE_PATH = "pretrained_model/db.npy"

    if args.w == 320:
        W, H = 320, 240
    else:
        W, H = 640, 640

    # load model
    if args.coral_tpu:
        interpreter = tflite.Interpreter(model_path='pretrained/retinaface_landmark_320_240_quant.tflite',
                                         experimental_delegates=[tflite.load_delegate('libedgetpu.1.dylib')])
    else:
        interpreter = tf.compat.v1.lite.Interpreter(model_path=MODEL_PATH)
        interpreter_rec = tf.compat.v1.lite.Interpreter(model_path=REC_MODEL_PATH)
    rec_db = np.load(DATABASE_PATH)

    # get handler
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    bbox_index = interpreter.get_output_details()[0]['index']
    ldmk_index = interpreter.get_output_details()[1]['index']
    prob_index = interpreter.get_output_details()[2]['index']
    interpreter_rec.allocate_tensors()
    rec_input_index = interpreter_rec.get_input_details()[0]['index']
    rec_output_index = interpreter_rec.get_output_details()[0]['index']


    # video capture
    cap = cv2.VideoCapture(0)

    while True:

        start = time.time()

        # read camera
        ret, frame = cap.read()

        # image processing
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
        rgb_f = (rgb / 255) - 0.5  # normalization

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
        if len(keep > 0):
            pred_bbox_pixel = pred_bbox_pixel[keep, :]
            pred_ldmk_pixel = pred_ldmk_pixel[keep, :]
            pred_prob = pred_prob[keep]

            # crop faces
            vaild_bboxs, face_imgs, face_landmarks = crop_faces(flipped, pred_bbox_pixel, pred_ldmk_pixel)
            # face recognition
            if len(face_imgs):
                aligned = face_algin_by_landmark(face_imgs[0], face_landmarks[0])
                aligned_norm = preprocess(aligned)

                interpreter_rec.set_tensor(rec_input_index, aligned_norm)
                interpreter_rec.invoke()
                # feature = rec_model.predict(aligned_norm)
                feature = interpreter_rec.get_tensor(rec_output_index)
                result = face_recognition(feature, rec_db)
                print(result)
                cv2.imshow('cropped', aligned)

            # draw result
            for i in (range(pred_prob.shape[0])):
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
    cap.release()
    cv2.destroyAllWindows()

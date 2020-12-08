import argparse
import json
import multiprocessing
import platform
import time

import cv2
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.tracker import Tracker
from FaceRecognizer import *
from FileVideoStreamer import *
from postprocessing import *

# FLAGS
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--platform', type=str,
                    help='picamera or desktop')
parser.add_argument('--w', type=int, default=320)
parser.add_argument('--coral_tpu', type=bool, default=False,
                    help="whether use tpu")
parser.add_argument('--mask', type=bool, default=False)
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

    MODEL_PATH_TPU = "../pretrained_model/edgetpu_v1/facedetection_320_240_edgetpu.tflite"
    REC_MODEL_PATH_TPU = "../pretrained_model/edgetpu_v2/model_with_mask_clf_quant_edgetpu.tflite"
    MODEL_PATH = "../pretrained_model/training_model/ulffd_landmark.tflite"
    REC_MODEL_PATH = "../pretrained_model/training_model/inference_model_993_quant.tflite"
    DATABASE_PATH = "../pretrained_model/db.npy"
    LABEL_PATH = "../pretrained_model/label.json"

    if args.w == 320:
        W, H = 320, 240
    else:
        W, H = 640, 640

    # load model
    if args.coral_tpu:
        face_detector = FaceDetector(MODEL_PATH_TPU, tpu=args.coral_tpu)
        face_recognizer = FaceRecognizer(REC_MODEL_PATH_TPU, tpu=args.coral_tpu, mask=args.mask)
    else:
        face_detector = FaceDetector(MODEL_PATH, tpu=args.coral_tpu)
        face_recognizer = FaceRecognizer(REC_MODEL_PATH, tpu=args.coral_tpu, mask=args.mask)

    # load database
    rec_db = np.load(DATABASE_PATH)
    label = json.load(open(LABEL_PATH))
    cap = FileVideoStream(0).start()

    # kalman filter
    metric = NearestNeighborDistanceMetric(
        "cosine", 0.2)
    ds_tracker = Tracker(metric, max_age=70)

    time.sleep(1)

    while True:

        start = time.time()

        # read camera
        ret, frame = cap.read()
        # flipped = q_flipped.get()
        # rgb_f = q_rgbf.get()

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
        rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB).astype('float32') #flipped[...,::-1].copy().astype('float32') #
        rgb_f = (rgb / 255) - 0.5  # normalization

        # detection
        pred_bbox_pixel, pred_ldmk_pixel, pred_prob = face_detector.detect_face(rgb_f)

        if len(pred_prob) > 0:

            # crop faces
            valid_index, vaild_bboxs, face_imgs, face_landmarks = crop_faces(flipped, pred_bbox_pixel, pred_ldmk_pixel)
            pred_bbox_pixel = pred_bbox_pixel[valid_index, :]
            pred_bbox_pixel_tlwh = pred_bbox_pixel.copy()
            pred_bbox_pixel_tlwh[:, 2:] = pred_bbox_pixel_tlwh[:, 2:] - pred_bbox_pixel_tlwh[:, :2]
            pred_ldmk_pixel = pred_ldmk_pixel[valid_index, :]
            pred_prob = pred_prob[valid_index]

            # loop over faces
            detections = []
            for i in (range(pred_prob.shape[0])):

                # face recognition
                aligned = face_algin_by_landmark(face_imgs[i], face_landmarks[i])
                feature, mask = face_recognizer.face_recognize(aligned)
                new_detection = Detection((pred_bbox_pixel_tlwh[i]), confidence=pred_prob[i], feature=feature.squeeze(0))
                detections.append(new_detection)
                result = face_recognition(feature, rec_db)

                if args.mask:
                    is_mask = np.argmax(mask) == 0
                    if is_mask:
                        mask_label = "mask"
                    else:
                        mask_label = "no mask"
                    ((label_width, label_height), _) = cv2.getTextSize(
                        mask_label,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        thickness=2
                    )
                    cv2.rectangle(
                        flipped,
                        (int((pred_bbox_pixel[i, 2])) - label_width, int(pred_bbox_pixel[i, 1])),
                        (int(int((pred_bbox_pixel[i, 2])) + label_width * 0.01),
                         int(pred_bbox_pixel[i, 1] + label_height + label_height * 0.25)),
                        color=(255, 0, 0),
                        thickness=cv2.FILLED
                    )
                    cv2.putText(flipped, mask_label,
                                (int((pred_bbox_pixel[i, 2])) - label_width,
                                 int(pred_bbox_pixel[i, 1] + label_height + label_height * 0.25)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                color=(255, 255, 255),
                                thickness=2
                                )

                # put label
                cv2.putText(flipped, label[str(result[0])],
                            (int((pred_bbox_pixel[i, 0])), int(pred_bbox_pixel[i, 1] - 10))
                            ,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # put rectangle
                cv2.rectangle(flipped, tuple(pred_bbox_pixel[i, :2].astype(int)),
                              tuple(pred_bbox_pixel[i, 2:].astype(int)), (255, 0, 0), 2)

                for ldx, ldy, color in zip(pred_ldmk_pixel[i][0::2].astype(int),
                                           pred_ldmk_pixel[i][1::2].astype(int),
                                           [(255, 0, 0), (0, 255, 0),
                                            (0, 0, 255), (255, 255, 0),
                                            (255, 0, 255)]):
                    cv2.circle(flipped, (ldx, ldy), 3, color, 2)

            ds_tracker.predict()
            ds_tracker.update(detections)
            for track in ds_tracker.tracks:
                cv2.rectangle(flipped, tuple(track.to_tlbr().astype(int)[:2]),
                              tuple(track.to_tlbr().astype(int)[2:]), (0, 0, 255), 2)
                # print(track.to_tlbr())

        cv2.imshow('detection', flipped)

        # print fps
        print("fps:", 1 / (time.time() - start))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # cam_process.terminate()
    # cam_process.join()
    cv2.destroyAllWindows()

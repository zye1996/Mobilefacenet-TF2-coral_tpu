import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from inference.FaceRecognizer import FaceRecognizer

# FLAGS
parser = argparse.ArgumentParser(description='Parsing args for generate database')
parser.add_argument("--images", type=str, help="face image path organized by names", required=True)
parser.add_argument("--db_dir", type=str, help="database output directory", default='.')
args = parser.parse_args()


if __name__ == '__main__':

    # model path
    REC_MODEL_PATH_TPU = "pretrained_model/mobilefacenet_edgetpu_cocompiled.tflite"
    recognizer = FaceRecognizer(REC_MODEL_PATH_TPU)

    # buffer
    label = []
    db = []

    for file in os.listdir(args.images):
        try:
            image = plt.imread(os.path.join(args.images, file))
            embedding = recognizer.face_recognize(image)
            label.append(file.split('.')[0])
            db.append(embedding)
        except:
            continue

    # write to file
    if len(label) > 0:
        np.save(os.path.join(args.db_dir, "label"), np.array(label))
        np.save(os.path.join(args.db_dir, "db"), np.array(db))

    print("saved %d embeddings" % len(label))

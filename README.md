# Face recognition with Coral EdgeTPU Support based on MobileFacenet

Mobilefacenet with Tensorflow-2, EdgeTPU models also supplied for running model on Coral EdgeTPU

## Introduction
Tensorflow 2 version of mobilefacenet from [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)

## Demo
[Video Link](https://www.youtube.com/watch?v=o6G-xXyHyAM)

![Alt Text](demo/demo.gif)

Running 60 fps on Desktop with Coral TPU, and around 24 fps on raspberry pi

## Usage

### Automatic Face Registering

#### Adding New Faces from an Image Folder
To register new faces, place images of each person in a folder organized by name:
```
images/
  yzy/
    yzy1.jpg
    yzy2.jpg
  steve/
    steve1.jpg
```
Then run the database generation script:
```bash
python utils/generate_db.py --images path/to/images --db_dir pretrained_model/
```
This will produce `db.npy` (face embeddings) and `label.json` (name-to-index mapping) in the output directory.

To merge new faces into an existing database, load the existing `db.npy` and `label.json`, append the new embeddings and labels, and save:
```python
import numpy as np
import json

old_db = np.load("pretrained_model/db.npy")
old_label = json.load(open("pretrained_model/label.json"))

new_db = np.load("pretrained_model/db.npy")  # from generate_db.py output
new_label = json.load(open("pretrained_model/label.json"))

merged_db = np.concatenate([old_db, new_db], axis=0)
merged_label = {**old_label, **new_label}

np.save("pretrained_model/db.npy", merged_db)
json.dump(merged_label, open("pretrained_model/label.json", "w"))
```

#### Removing Faces from the Database
To remove a face, delete its entry from `label.json` and the corresponding row from `db.npy`:
```python
import numpy as np
import json

db = np.load("pretrained_model/db.npy")
label = json.load(open("pretrained_model/label.json"))

# Remove a specific label
idx_to_remove = label.pop("steve", None)
if idx_to_remove is not None:
    db = np.delete(db, idx_to_remove, axis=0)
    # Re-index remaining labels
    label = {name: i for i, name in enumerate(label)}

np.save("pretrained_model/db.npy", db)
json.dump(label, open("pretrained_model/label.json", "w"))
```

#### Hot-Saving and Retraining New Faces from Camera
During inference, captured face embeddings can be appended to the database in real time:
```python
# After detecting and recognizing a face, save the new embedding
new_feature = feature  # from face_recognizer.face_recognize()
db = np.load("pretrained_model/db.npy")
label = json.load(open("pretrained_model/label.json"))

new_id = max(label.values()) + 1 if label else 0
label[str(new_id)] = "new_person_name"
db = np.concatenate([db, new_feature.reshape(1, -1)], axis=0)

np.save("pretrained_model/db.npy", db)
json.dump(label, open("pretrained_model/label.json", "w"))
```
To make the new face persist across sessions, retrain the recognition model using the updated database.

### Dataset
Use the same dataset as used in [Mobilefacenet-Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) to train. 
CASIA is used for training and LFW is used for testing.

### Training
Change the directory pointing to image dataset in train.py. I trained the model directly with ArcFace by setting RESUME 
to False but it is worthwhile to try out pretraining with softmax loss

I added an example to add extra header to perform classification using generated embedding, here I use generated embedding 
to make prediction on whether a person is wearing mask. You can have more fun by using another dataset 

### Result
Trained model is evaluate on each epoch use LFW dataset and I got 99.3% accuracy without pretraining

# Credit to
1. Playground by Qihang Zheng: https://github.com/zhen8838/playground
2. Mobilefacenet-Pytorch: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
3. MobileFaceNet-Keras: https://github.com/TMaysGGS/MobileFaceNet-Keras

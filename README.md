# Face recognition with Coral EdgeTPU Support based on MobileFacenet

Mobilefacenet with Tensorflow-2, EdgeTPU models also supplied for running model on Coral EdgeTPU

## Introduction
Tensorflow 2 version of mobilefacenet from [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)

## Demo
[Video Link](https://www.youtube.com/watch?v=o6G-xXyHyAM)

![Alt Text](demo/demo.gif)

## Usage

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

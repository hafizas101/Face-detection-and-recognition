# Face-detection-and-recognition
The notebook "face_rec3.ipynb" has been developed according to the instructions present in "FACE_PERSON RECOGNITION MODULE.pdf" file. It mainly works on the principle of transfer learning and triplet loss network. Detailed explanation is available on [pyimagesearch](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/). 
<p align="center">
  <img width="500" height="300" src="https://github.com/hafizas101/Face_recognition/blob/master/results/labelled_frames/00007.jpg">
</p>
It consists of two main modules:

### Extracting faces fromtraining images
After running first and second cell, we can run module5 cell to extract faces from all training images present in dataset directory.

### Face recognition
First of all, we extract frames from our input video. Then, we load training faces and convert each face into 128 dimensional embedding vector and stack all training faces embeddings with their corresponding person name label. Similarly we extract 128 dimensional embedding vector of test face and calculate distance between the test face embedding vector and traing face embeddings. If this distance is less than a threshold e.g 0.6, the two faces belong to the same person otherwise they are different people. Then finally, we label the names on each face and save the results in the form of a video and csv files.

# face-mask-detection
Build a real-time system to detect whether the person on the webcam is wearing a mask or not. Train the face mask detector by deep learning model using Keras, OpenCV and MobileNet.

Abstract

Coronavirus disease 2019 has affected the world seriously. One major protection method for people is to wear masks in public areas. Furthermore, many public service providers require customers to use the service only if they wear masks correctly. However, there are only a few research studies about face mask detection based on image analysis. In this assignment, we propose Face Mask Detection, which is a high-accuracy and efficient face mask detector. The proposed Face Mask Detector is a one-stage detector, which consists of a feature pyramid network to fuse high-level semantic information with multiple feature maps, and a novel context attention module to focus on detecting face masks. Besides, we also implement it with a light-weighted neural network MobileNet for embedded or mobile devices.

During pandemic COVID-19, WHO has made wearing masks compulsory to protect against this deadly virus. In this assignment we developed a machine learning project – Real-time Face Mask Detector with Python.
 
Introduction
In December 2019 a virus was found which caused a pandemic starting from Wuhan, China and is came to known as Coronavirus or Covid-19. In matter of time everybody was advised as well as order to use the mask for their own safety. Government also started punishing those not wearing mask by collecting fine. This project is helpful in such scenarios where it can detect face mask in real time. In can be used in an embedded system to detect face mask. Such devices can be installed at traffic signals and other public places.

Objectives

We will build a real-time system to detect whether the person on the webcam is wearing a mask or not. We will train the face mask detector model using Keras and OpenCV.
•	Real-time face mask detection using Python, OpenCV, Keras and MobileNet.
•	Preprocessing of data and calculating the probability of with and without mask.
•	Plotting training loss and accuracy.

Dataset
The dataset we are working on consists of 3833 images with 1915 images containing images of people wearing masks and 1918 images with people without masks.
Link for Kaggle Data set: https://www.kaggle.com/omkargurav/face-mask-dataset
 
Face Mask Detection
First of all, we need to install all the requirements in the project and are given in requirements.txt file. This can be done by going to the location of the file and giving the command:
Pip install -r requirements.txt
After installation we can go to the further stages:
Libraries we used:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2


Data Preprocessing:
Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model. In this project we converted all the images into arrays. 

We created two list ‘data’ and ‘labels’ and we add the image to data list after processing and its corresponding category (with mask or without mask) to labels. But we have a problem here, the labels are alphabetical. So, we converted them to binary form by LabelBinarizer() function which comes from sklearn.preprocessing.




lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

Now, we converted lists to numpy arrays.

data = np.array(data, dtype="float32")
labels = np.array(labels)

After this we need to split the training and testing data using train_test_split function. Here we gave test_size as 0.20 which means 20% of the images are given for testing while 80% are given for training.

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)



MobileNets:
Instead of normal convolution (in below image) we are going to use Mobile Nets as they are very fast compared to convolutional neural network and Mobile Nets uses lesser parameters. Then we will do Max-pooling. After that we will flatten it then we will create a fully connected layer and finally output.
 



 
Image Data Generator: 
This function is used to create many images with single image by adding various properties. So, we can create more data set with this.

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

Modelling Part: 
We created two models ‘base model’ and ‘head model’. Output of base model is given input to the head model. 
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

Train the head of network:

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

Plot the training loss and accuracy:
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
 
We can see accuracy is going on increasing and loss is going down.

•	Now we can run the train_mask_dectector.py file through command prompt. After running the file, it will take sometime say 20-30 mins to train the model. After that we can move to the part where we will make use of model in real time.

•	Apply the model in camera: We don’t have a face detector model. So, we downloaded two files in face detector folder which is used in face detection.

Now, for camera operations we are going to use OpenCV.

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

This faceNet will be used to detect the face.

maskNet = load_model("mask_detector.model")

maskNet will be used to detect mask. We loaded our model using load_model.


vs = VideoStream(src=0).start()

This will load our camera. Here, src=0 indicates we are using aur primary camera.


Now in while loop we are going to consider every frame in video as a image and predict the mask or without mask percentage.

frame = vs.read()
frame = imutils.resize(frame, width=400)

We defined a function called detect_and_predict_mask() which returns location of mask and prediction of with mask and without mask.

(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

Now, we need to label the output if it masked or not. We also specified color for every case. Green for with mask and red for without.

label = "Mask" if mask > withoutMask else "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

We need to insert text to our frame of with mask and without mask. We also need to specify the prediction. We specified the font style as ‘Hershey simplex’ also the coordinates of text and the rectangle.

label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

cv2.putText(frame, label, (startX, startY - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


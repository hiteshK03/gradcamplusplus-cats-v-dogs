# Usage: python3 apply_gradcam.py -i ./dog.jpg -m my_model.h5

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from gradcam_utils import grad_cam, grad_cam_plus
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help = "Model to be used")
ap.add_argument("-i", "--image_path", type=str, required=True, 
	help = "Path of the image")
args = vars(ap.parse_args())

img = args['image_path']
orig_image = load_img(img, target_size=(150, 150))
image = img_to_array(orig_image)
image = np.expand_dims(image, axis=0)
image = image/255.0

model = load_model(args['model'])
# Apply gradcam & gradcam++ visualisations
gradcam=grad_cam(model,image,layer_name='mixed7')
gradcamplus=grad_cam_plus(model,image,layer_name='mixed7')

# Decode predictions and probabilities
print("IMG : ", img)
prob = model.predict(image)
pred_label = np.round(prob[0][0]).astype(int)
if pred_label == 0:
  prob[0][0] = 1-prob[0][0]

labels = {'cats': 0, 'dogs': 1}
labels = dict((v,k) for k,v in labels.items())
print("class activation map for:",labels[pred_label])
print("Probability : ", prob[0][0])

fig, ax = plt.subplots(nrows=1,ncols=3)
plt.subplot(131)
plt.imshow(orig_image)
plt.title("input image")
plt.subplot(132)
plt.imshow(orig_image)
plt.imshow(gradcam,alpha=0.7,cmap="jet")
plt.title("Grad-CAM")
plt.subplot(133)
plt.imshow(orig_image)
plt.imshow(gradcamplus,alpha=0.7,cmap="jet")
plt.title("Grad-CAM++")
plt.show()
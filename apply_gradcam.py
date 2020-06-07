import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from gradcam_utils import grad_cam, grad_cam_plus

path = "/content/preds/"
img = "test/55.jpg"
image_path = os.path.join(path, img)
orig_image = load_img(image_path, target_size=(150, 150))
image = img_to_array(orig_image)
image = np.expand_dims(image, axis=0)
image = image/255.0

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
plt.imshow(gradcam,alpha=0.8,cmap="jet")
plt.title("Grad-CAM")
plt.subplot(133)
plt.imshow(orig_image)
plt.imshow(gradcamplus,alpha=0.8,cmap="jet")
plt.title("Grad-CAM++")
plt.show()
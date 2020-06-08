import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from model import *
from dataloader import *

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "./checkpoint/"

#Prepare call backs
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
LR_callback = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=2, factor=.5, min_lr=.00001)
EarlyStop_callback = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

my_callback=[EarlyStop_callback, LR_callback, tensorboard_callback, cp_callback]

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

# Train the model
history = model.fit(train_generator,epochs=3,verbose=2,validation_data=validation_generator)

# Generate plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

model.save('my_model.h5')

pred=model.predict(test_generator,verbose=1)
cl=np.round(pred)
cl=np.squeeze(cl, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in cl]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from model import *
from dataloader import *

# def train(model, train_data, validation_data, epochs, lr)

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

history = model.fit(train_generator,epochs=3,verbose=2,validation_data=validation_generator)

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

pred=model.predict(test_generator,verbose=1)
cl=np.round(pred)
cl=np.squeeze(cl, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in cl]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
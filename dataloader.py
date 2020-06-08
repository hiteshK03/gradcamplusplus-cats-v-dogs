import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./cats-v-dogs/train/"
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,  
        target_size=(150, 150),
        class_mode='binary')

VALIDATION_DIR = "./cats-v-dogs/validation/"
validation_datagen = ImageDataGenerator(rescale=1./255)

# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

TEST_DIR = "./cats-v-dogs/testing/"
test_datagen = ImageDataGenerator(rescale=1./255)

# TEST GENERATOR.
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=10,
    class_mode=None,
    shuffle=False
)
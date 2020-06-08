import os
import zipfile
import shutil
from shutil import copyfile, rmtree
import random

path_cats_and_dogs = "./dogs-vs-cats-redux-kernels-edition.zip"
path_train = "./train.zip"
path_test = "./test.zip"

zip_ref = zipfile.ZipFile(path_cats_and_dogs, 'r')
zip_ref.extractall('./')
zip_ref.close()
print("Extracted dogs-vs-cats-redux-kernels-edition.zip")

zip_ref = zipfile.ZipFile(path_train, 'r')
zip_ref.extractall('./')
zip_ref.close()
print("Extracted train.zip")

try:
	for i in ['dogs', 'cats']:
		os.mkdir('./' + i)
except OSError:
	pass

source = "./train/"
cat_dir = "./cats/"
dog_dir = "./dogs/"

for file in os.listdir(source):
	if 'cat' in file:
		copyfile(source+file, cat_dir+file)
	elif 'dog' in file:
		copyfile(source+file, dog_dir+file)
print("Copying done!!!")

try:
	dirs_list = ['./cats-v-dogs/','./cats-v-dogs/train/',"./cats-v-dogs/train/cats/",
				"./cats-v-dogs/train/dogs/",'./cats-v-dogs/validation/',
				"./cats-v-dogs/validation/cats/","./cats-v-dogs/validation/dogs/",'./cats-v-dogs/testing/']
	for dirs in dirs_list:
		os.mkdir(dirs)
except OSError:
	pass

zip_ref = zipfile.ZipFile(path_test, 'r')
zip_ref.extractall('./cats-v-dogs/testing/')
zip_ref.close()
print("Extracted test.zip")

def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    list_rand = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
    for file in list_rand[:int(SPLIT_SIZE*len(os.listdir(SOURCE)))]:
        if os.path.getsize(SOURCE+file) != 0:
            copyfile(SOURCE+file, TRAINING+file)
    for file in list_rand[int(SPLIT_SIZE*len(os.listdir(SOURCE))):]:
        if os.path.getsize(SOURCE+file) != 0:
            copyfile(SOURCE+file, VALIDATION+file)


TRAINING_CATS_DIR = "./cats-v-dogs/train/cats/"
VALIDATION_CATS_DIR = "./cats-v-dogs/validation/cats/"

TRAINING_DOGS_DIR = "./cats-v-dogs/train/dogs/"
VALIDATION_DOGS_DIR = "./cats-v-dogs/validation/dogs/"

split_size = 0.9
split_data(cat_dir, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(dog_dir, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)
print("Working directory made")

print("Cleaning up unnecessary folders")
rmtree("./train.zip")
rmtree("./test.zip")
rmtree(source)
rmtree(cat_dir)
rmtree(dog_dir)
print("All tasks done!!!!")
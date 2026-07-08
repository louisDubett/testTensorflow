#based on https://www.tensorflow.org/tutorials/images/classification?hl=fr
import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

def fill_class_names(data_dir):
	samples = []
	labels = []
	class_names = []
	class_index = 0
	for dirname in sorted(os.listdir(data_dir)):
		class_names.append(dirname)
		dirpath = data_dir / dirname
		fnames = os.listdir(dirpath)
		print("Processing %s, %d files found" % (dirname, len(fnames)))
		for fname in fnames:
			fpath = dirpath / fname
			f = open(fpath, encoding="latin-1")
			content = f.read()
			lines = content.split("\n")
			lines = lines[10:]
			content = "\n".join(lines)
			samples.append(content)
			labels.append(class_index)
		class_index += 1

	print("Classes:", class_names)
	print("Number of samples:", len(samples))

def train(dataset_url):
	#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

	print(f"training for {dataset_url}")
	data_dir = tf.keras.utils.get_file('chimie-dataset', origin=dataset_url, untar=True)
	data_dir = pathlib.Path(data_dir)
	# sub dir as in example from google
	data_dir = data_dir / "DATASET"
	if not os.path.exists(data_dir):
		raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")
	image_count = len(list(data_dir.glob('*/*.jpg')))
	print(f"Total images found: {image_count}")
	if image_count == 0:
		print("Warning: No images found. Check your dataset path and format.")
		all_files = list(data_dir.glob('*/*'))
		print(f"Found files (first 5): {[str(f) for f in all_files[:5]]}")

	#training split
	batch_size = 32
	img_height = 300
	img_width = 400

	#We will split the dataset into 80% training and 20% validation datasets.
	#Training Split: Data on which the model trains on.
	#    seed=123,image_size=(180, 180), batch_size=32): Sets a fixed random seed, target size and batch size respectively
	# If labels is "inferred", it should contain subdirectories, each containing images for a class. Otherwise, the directory structure is ignored.
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		labels="inferred",
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	#validation split
	val_ds = tf.keras.utils.image_dataset_from_directory(
	  data_dir,
	  labels="inferred",
	  validation_split=0.2,
	  subset="validation",
	  seed=123,
	  image_size=(img_height, img_width),
	  batch_size=batch_size)

	#see Configurer l'ensemble de données pour les performance
	#https://www.tensorflow.org/tutorials/images/classification?hl=fr
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	image_batch, label_batch = next(iter(train_ds))

	#train_ds.class_names does not exist
	#class_names = train_ds.class_names
	#print("list of training classes "+class_names)

	#https://www.tensorflow.org/tutorials/images/classification?hl=fr#standardize_the_data
	normalization_layer = layers.Rescaling(1./255)
	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	#image_batch, labels_batch = next(iter(normalized_ds))
	first_image = image_batch[0]
	# Notice the pixel values are now in `[0,1]`.
	print(np.min(first_image), np.max(first_image))

	#visualize dataset
	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(25):
			ax = plt.subplot(5, 5, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
	#		plt.title(class_names[labels[i]])
			plt.axis("off")

	#Building the Model
	#see https://www.geeksforgeeks.org/python/image-recognition-using-tensorflow/
	#Here we design CNN (Convolutional Neural Network) model using Keras Sequential() model which is commonly used model. We will use three convolution layers with Conv2D and MaxPooling2D followed by a dense layer to classify images.
	#    layers.Rescaling(1./255, input_shape=(180,180, 3)): Rescales images to [0,1] and sets input image size.
	#    layers.Conv2D(16, 3, padding='same', activation='relu'): Adds a convolutional layer with 16 filters and ReLU activation.
	#    layers.MaxPooling2D(): Adds a max-pooling layer to down sample feature maps.
	num_classes = 6 # len(class_names)
	#TODO fix this hard coded value to be dynamic based on the dataset
	
	model = Sequential([
		layers.Rescaling(1./255, input_shape=(img_height,img_width, 3)),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(num_classes)
	])

	#compiling the model
	model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(
					from_logits=True),
				metrics=['accuracy'])
	model.summary()

	#train
	epochs=10
	history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs
	)

	#visualizing training
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs_range = range(epochs)
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()

	#https://www.tensorflow.org/tutorials/images/classification?hl=fr#data_augmentation
	data_augmentation = keras.Sequential(
	  [
		layers.RandomFlip("horizontal",
						  input_shape=(img_height,
									  img_width,
									  3)),
		layers.RandomRotation(0.1),
		layers.RandomZoom(0.1),
	  ]
	)

	#https://www.tensorflow.org/tutorials/images/classification?hl=fr#dropout
	model = Sequential([
	  data_augmentation,
	  layers.Rescaling(1./255),
	  layers.Conv2D(16, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(32, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(64, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Dropout(0.2),
	  layers.Flatten(),
	  layers.Dense(128, activation='relu'),
	  layers.Dense(num_classes)
	])

	model.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])

	epochs = 15
	history = model.fit(
	  train_ds,
	  validation_data=val_ds,
	  epochs=epochs
	)

	#https://www.tensorflow.org/tutorials/images/classification?hl=fr#visualize_training_results_2
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()

	#test
	sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
	sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

	img = tf.keras.utils.load_img(
		sunflower_path, target_size=(img_height, img_width)
	)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	# The '.h5' extension indicates that the model should be saved to HDF5.
	model.save('my_model.keras')

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	#class_names=["daisy","dandelion","rose","sunflowers","tulips"]
	class_names=["becher","erlenmeyer","kolben","messzylinder","pipette","reagenzglas"]
	print(
		"This image most likely belongs to {} with a {:.2f} percent confidence."
		.format(class_names[np.argmax(score)], 100 * np.max(score))
	)

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description='Image Classification from tensorflow')
	parser.add_argument("-dataset_url")
	parser.add_argument('-test',	action='store_true')
	try:
		args = parser.parse_args()
		if (args.test):
			#train("/home/christian/Téléchargements/flower_photos2.tgz")
			train("file:/home/christian/git/testTensorflow/chimie-dataset.tgz")
		else:
			if args.dataset_url is None:
				print("missing image argument")
				train("file:/home/christian/git/testTensorflow/chimie-dataset.tgz")
				parser.print_help()
			else:
				print('processing '+ args.dataset_url)
				train(args.dataset_url)
	except (argparse.ArgumentError or argparse.ArgumentTypeError):
		print("failed to parse arguments")
		parser.print_help()

	print("TensorFlow version:", tf.__version__)

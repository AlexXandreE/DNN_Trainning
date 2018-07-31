import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

def dnn_model():
	model = keras.Sequential()
	# Adds a densely-connected layer with 64 units to the model:
	model.add(keras.layers.Dense(784, activation='relu'))
	# Add another:
	model.add(keras.layers.Dense(100, activation='sigmoid'))
	# Add a softmax layer with 10 output units:
	model.add(keras.layers.Dense(10, activation='softmax'))

	return model

trainning_data = pd.read_csv("train.csv", sep=",")
labels = trainning_data['label'].values
labels = keras.utils.to_categorical(labels)
features = trainning_data.drop("label", axis=1).values / 255

model = dnn_model()

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(features, labels, epochs=10, batch_size=32)

testing_data = pd.read_csv("test.csv", sep=",")

features = testing_data.values / 255
result = model.predict_classes(features)
with open("sub_1.csv", "w") as out:
	i = 1
	out.write("ImageId,Label\n")
	for value in result:
		out.write(str(i) + "," + str(value) + "\n")
		i += 1

print(result)
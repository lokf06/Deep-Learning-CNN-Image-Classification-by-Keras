# Imports

import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
from cnn_utils import run_architecture
from sklearn.metrics import classification_report

# Main

K.set_image_data_format('channels_last')

seed = 88
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])

run_architecture('A1', model, X_train, y_train, X_test, y_test)

# Train the model
# epochs=10: process entire dataset 10 times
# batch_size=200: update weights after every 200 samples
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# Predict the classes for test data, use argmax to convert the one-hot encoded prediction vector into an integer class label
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Convert y_test from one-hot encoded format to class labels
y_test_labels = np.argmax(y_test, axis=-1)

# Generate and display classification report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=[str(i) for i in range(num_classes)]))
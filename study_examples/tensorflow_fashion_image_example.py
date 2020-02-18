import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255
test_images = test_images / 255

class_names = ['Tshirt',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']



# print(type(train_images[0]))
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

amodel = keras.Sequential([  # sequential means a sequence of layers
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),     # dense=> fully connected,
    keras.layers.Dense(10, activation="softmax")])  # softmax, this layer add to one, means probability

amodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

amodel.fit(train_images, train_labels, epochs=5)
# epoch: how many times model sees the same image
# give more epochs does not mean more accurate

test_loss, test_acc = amodel.evaluate(test_images, test_labels)

print(f'test acc is {test_acc}')

# use models to predict ..
# how to use the model..
prediction = amodel.predict(test_images)


print(f'I predict it is a {class_names[np.argmax(prediction[0])]}')


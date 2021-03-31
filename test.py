import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np

model = load_model('cifar10_model.h5')

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(test_samples, test_labels) = datasets.cifar10.load_data()[1]

test_labels = test_labels.reshape(-1,)
test_samples = test_samples / 255.0

predictions = model.predict(test_samples)


def plot_sample(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.show()


y_classes = [np.argmax(element) for element in predictions]

for i in range(130, 140):
    plot_sample(test_samples, test_labels, i)
    print(classes[y_classes[i]])





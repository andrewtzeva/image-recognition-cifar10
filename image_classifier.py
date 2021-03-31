from tensorflow.keras import datasets, layers, models

# Loading images
(train_samples, train_labels), (test_samples, test_labels) = datasets.cifar10.load_data()

# Reshaping label arrays
train_labels = train_labels.reshape(-1,)
test_labels = test_labels.reshape(-1,)

# Normalizing each pixel - [0,1]
train_samples = train_samples / 255.0
test_samples = test_samples / 255.0

# Building the CNN
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_samples, train_labels, epochs=10)

# Evaluation
model.evaluate(test_samples, test_labels)

# Save
model.save('cifar10_model.h5')







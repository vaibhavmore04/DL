
# ASSIGNMENT 5: Comparison of MLP and CNN on MNIST Dataset

# STEP 1: Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# Import neural network models and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, RandomRotation, RandomZoom, Input, Dropout

# Import activation functions and regularization
from tensorflow.keras.activations import relu, tanh, softmax
from tensorflow.keras.regularizers import l2


# STEP 2: Load MNIST dataset

# Read training and testing CSV files
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

# Separate image pixels and labels
X_train = train.drop('label', axis=1).values
Y_train = train['label'].values

X_test = test.drop('label', axis=1).values
Y_test = test['label'].values

# Convert pixel data into 28x28 images
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0


# STEP 3: Create MLP Model
# MLP means Multilayer Perceptron

model_mlp = Sequential([

    # Convert 28x28 image into 784 values
    Flatten(input_shape=(28, 28)),

    # First hidden layer with 256 neurons
    Dense(256, activation='relu'),

    # Second hidden layer with 128 neurons
    Dense(128, activation='tanh'),

    # Third hidden layer with 64 neurons
    Dense(64, activation='tanh'),

    # Output layer for digits 0-9
    Dense(10, activation='softmax')
])


# STEP 4: Compile MLP Model

model_mlp.compile(
    optimizer='adagrad',                    # Optimizer
    loss='sparse_categorical_crossentropy', # Loss function
    metrics=['accuracy']                    # Accuracy
)

# Display MLP structure
model_mlp.summary()


# STEP 5: Create Data Augmentation
# Data augmentation creates modified images during training

data_augmentation = Sequential([

    # Randomly rotate images
    RandomRotation(0.1),

    # Randomly zoom images
    RandomZoom(0.1)
])


# STEP 6: Create CNN Model
# CNN means Convolutional Neural Network

model_cnn = Sequential([

    # Input image size: 28x28 with 1 channel
    Input(shape=(28, 28, 1)),

    # Apply random rotation and zoom
    data_augmentation,

    # First convolution layer: 32 filters
    Conv2D(32, kernel_size=(3, 3), activation='relu'),

    # Reduce image size
    MaxPooling2D(pool_size=(2, 2)),

    # Second convolution layer: 64 filters
    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # Again reduce image size
    MaxPooling2D(pool_size=(2, 2)),

    # Convert feature maps into 1D
    Flatten(),

    # Fully connected layer with 128 neurons
    Dense(128, activation='relu'),

    # Randomly disable 50% neurons during training
    Dropout(0.5),

    # Output layer for 10 digits
    Dense(10, activation='softmax')
])


# STEP 7: Compile CNN Model

model_cnn.compile(
    optimizer='adagrad',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display CNN structure
model_cnn.summary()


# STEP 8: Compile Both Models Using Adam Optimizer
# Adam replaces Adagrad before training in original notebook

model_mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_mlp.summary()

model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_cnn.summary()


# STEP 9: Train MLP Model

history_mlp = model_mlp.fit(
    X_train,
    Y_train,
    epochs=10,             # Train for 10 epochs
    validation_split=0.2   # 20% validation data
)


# STEP 10: Train CNN Model

history_cnn = model_cnn.fit(
    X_train,
    Y_train,
    epochs=10,
    validation_split=0.2
)


# STEP 11: Plot MLP vs CNN Accuracy Graph

plt.figure(figsize=(10, 6))

# MLP training accuracy
plt.plot(
    history_mlp.history["accuracy"],
    label="MLP Training"
)

# MLP validation accuracy
plt.plot(
    history_mlp.history["val_accuracy"],
    linestyle="--",
    label="MLP Validation"
)

# CNN training accuracy
plt.plot(
    history_cnn.history["accuracy"],
    label="CNN Training"
)

# CNN validation accuracy
plt.plot(
    history_cnn.history["val_accuracy"],
    linestyle="--",
    label="CNN Validation"
)

# Graph title and labels
plt.title("MLP vs CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.show()


# STEP 12: Plot MLP vs CNN Loss Graph

plt.figure(figsize=(10, 6))

# MLP training loss
plt.plot(
    history_mlp.history["loss"],
    label="MLP Training"
)

# MLP validation loss
plt.plot(
    history_mlp.history["val_loss"],
    linestyle="--",
    label="MLP Validation"
)

# CNN training loss
plt.plot(
    history_cnn.history["loss"],
    label="CNN Training"
)

# CNN validation loss
plt.plot(
    history_cnn.history["val_loss"],
    linestyle="--",
    label="CNN Validation"
)

# Graph title and labels
plt.title("MLP vs CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.show()


# STEP 13: Predict First 10 Test Images

# CNN predicts digits for first 10 images
predictions = model_cnn.predict(X_test[:10])

# Select class having highest probability
predicted_classes = np.argmax(
    predictions,
    axis=1
)


# STEP 14: Display Actual and Predicted Digits

plt.figure(figsize=(12, 5))

# Display 10 handwritten digit images
for i in range(10):

    # Arrange images in 2 rows and 5 columns
    plt.subplot(2, 5, i+1)

    # Display digit in grayscale
    plt.imshow(X_test[i], cmap='gray')

    # Show actual and predicted digit
    plt.title(f"Actual: {Y_test[i]}\nPred: {predicted_classes[i]}")

    # Hide axes
    plt.axis("off")

# Display all images
plt.show()

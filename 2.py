
# ASSIGNMENT 2: Build and Train a Feedforward Neural Network

# STEP 1: Import required libraries

import pandas as pd                    # To read and handle dataset
import matplotlib.pyplot as plt         # To plot accuracy graph
import tensorflow as tf                 # Deep learning library

from tensorflow.keras.models import Sequential  # To create neural network
from tensorflow.keras.layers import Dense,Flatten,Input  # Neural network layers
from tensorflow.keras.activations import relu,tanh,softmax  # Activation functions


# STEP 2: Load and preprocess MNIST dataset

# Read training and testing CSV files
train=pd.read_csv('mnist_train.csv')
test=pd.read_csv('mnist_test.csv')

# Display first 5 rows of training dataset
train.head()

# Separate input pixels (X) and output labels (Y)
X_train=train.drop('label',axis=1).values
Y_train=train['label'].values

X_test=test.drop('label',axis=1).values
Y_test=test['label'].values

# Convert pixel data into 28x28 image format
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)

# Normalize pixel values from 0-255 to 0-1
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0


# STEP 3: Build Feedforward Neural Network (MLP)

model=Sequential([

    # Flatten converts 28x28 image into 784 input values
    Flatten(input_shape=(28,28)),

    # First hidden layer: 256 neurons with ReLU
    Dense(256,activation='relu'),

    # Second hidden layer: 128 neurons with Tanh
    Dense(128,activation='tanh'),

    # Third hidden layer: 64 neurons with Tanh
    Dense(64,activation='tanh'),

    # Output layer: 10 neurons for digits 0 to 9
    Dense(10,activation='softmax')
])


# STEP 4: Compile the model

model.compile(

    # Adagrad optimizer updates weights to reduce error
    optimizer='adagrad',

    # Loss function calculates prediction error
    loss='sparse_categorical_crossentropy',

    # Calculate model accuracy
    metrics=['accuracy']
)

# Display model architecture
model.summary()


# STEP 5: Train the model

history=model.fit(

    # Training images and their labels
    X_train,Y_train,

    # Train for 10 epochs
    epochs=10,

    # Use 20% training data for validation
    validation_split=0.2
)


# STEP 6: Evaluate model performance

# Test model using testing dataset
loss,accuracy=model.evaluate(X_test,Y_test)

# Display test loss
print(loss)

# Display test accuracy
print(accuracy)


# STEP 7: Plot Accuracy Graph

# Plot training accuracy
plt.plot(history.history['accuracy'],label="Accuracy")

# Plot validation accuracy
plt.plot(history.history['val_accuracy'],label="Validation Acuuracy")

# Set graph title
plt.title('Accuracy and Validation Accuracy')

# Display graph labels
plt.legend()

# X-axis shows epochs
plt.xlabel('epochs')

# Y-axis shows accuracy
plt.ylabel('accuracy')

# Display graph
plt.show()


# ASSIGNMENT 4: Comparison of Activation Functions and Gradients

# STEP 1: Import required libraries

import pandas as pd                  # To handle dataset
import tensorflow as tf              # Deep learning library
import matplotlib.pyplot as plt      # To plot graphs

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
import time                          # To calculate training time


# STEP 2: Load Fashion-MNIST dataset

# Read training and testing CSV files
data_train = pd.read_csv(
    r"C:\Users\Sayali Jadhav\Downloads\Datasetdl\Dataset\Fashion\fashion-mnist_train.csv"
)

data_test = pd.read_csv(
    r"C:\Users\Sayali Jadhav\Downloads\Datasetdl\Dataset\Fashion\fashion-mnist_test.csv"
)


# STEP 3: Separate input pixels and output labels

# Training images and labels
x_train = data_train.drop("label", axis=1).values
y_train = data_train["label"].values

# Testing images and labels
x_test = data_test.drop("label", axis=1).values
y_test = data_test["label"].values


# STEP 4: Normalize dataset

# Convert pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


# STEP 5: Reshape images

# Convert 784 pixels into 28x28 images
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)


# STEP 6: Create and train neural network

# Function to create models with different activation functions
def create_model(activation):

    # Create Sequential neural network
    model = Sequential([

        # Convert 28x28 image into 784 values
        Flatten(input_shape=(28, 28)),

        # First hidden layer with 128 neurons
        Dense(128, activation=activation),

        # Second hidden layer with 64 neurons
        Dense(64, activation=activation),

        # Third hidden layer with 32 neurons
        Dense(32, activation=activation),

        # Output layer for 10 fashion categories
        Dense(10, activation="softmax")
    ])

    # Compile model using Adam optimizer
    model.compile(
        optimizer=Adam(),
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    # Record training start time
    start = time.time()

    # Train model for 5 epochs
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test)
    )

    # Record training end time
    end = time.time()

    # Evaluate model on testing dataset
    test_acc, test_loss = model.evaluate(
        x_test, y_test, verbose=0
    )

    # Calculate total training time
    training_time = end - start

    # Return model and results
    return model, history, test_acc, test_loss, training_time


# STEP 7: Train ReLU Model

# ReLU returns 0 for negative inputs and x for positive inputs
relu_model, relu_history, relu_loss, relu_acc, relu_time = create_model("relu")


# STEP 8: Train Tanh Model

# Tanh gives output between -1 and 1
tanh_model, tanh_history, tanh_loss, tanh_acc, tanh_time = create_model("tanh")


# STEP 9: Train Sigmoid Model

# Sigmoid gives output between 0 and 1
sigmoid_model, sigmoid_history, sigmoid_loss, sigmoid_acc, sigmoid_time = create_model("sigmoid")


# STEP 10: Calculate gradients using GradientTape

# Gradients show how loss changes with model weights
def calculate_gradients(model, x_train, y_train):

    # Select first 32 training samples
    x_sample = x_train[:32]
    y_sample = y_train[:32]

    # Record operations for gradient calculation
    with tf.GradientTape() as tape:

        # Predict output using model
        predictions = model(x_sample, training=True)

        # Calculate prediction loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_sample, predictions
        )

        # Calculate average loss
        loss = tf.reduce_mean(loss)

    # Calculate gradients for trainable weights
    gradients = tape.gradient(
        loss, model.trainable_variables
    )

    # Display loss value
    print("Loss:", loss.numpy())

    # Display gradient details for each weight
    for variable, gradient in zip(
        model.trainable_variables, gradients
    ):
        print(variable.name)
        print("Gradient shape:", gradient.shape)
        print("Mean gradient:", tf.reduce_mean(gradient).numpy())
        print("Gradient norm:", tf.norm(gradient).numpy())
        print("------------------------")

    return gradients


# STEP 11: Calculate gradients for all three models

# ReLU gradients
relu_gradients = calculate_gradients(relu_model, x_train, y_train)

# Tanh gradients
tanh_gradients = calculate_gradients(tanh_model, x_train, y_train)

# Sigmoid gradients
sigmoid_gradients = calculate_gradients(sigmoid_model, x_train, y_train)


# STEP 12: Calculate average gradient values

def average_gradient(gradients):

    # Calculate average absolute value of gradients
    values = [
        tf.reduce_mean(tf.abs(g)).numpy()
        for g in gradients if g is not None
    ]

    # Return overall average gradient
    return sum(values) / len(values)


# Print average gradients for comparison
print("ReLU Gradient:", average_gradient(relu_gradients))
print("Tanh Gradient:", average_gradient(tanh_gradients))
print("Sigmoid Gradient:", average_gradient(sigmoid_gradients))


# STEP 13: Plot Average Gradient Comparison

# Names of activation functions
names = ['ReLU', 'Tanh', 'Sigmoid']

# Average gradient values
values = [
    average_gradient(relu_gradients),
    average_gradient(tanh_gradients),
    average_gradient(sigmoid_gradients)
]

# Create bar graph
plt.figure(figsize=(8, 5))
plt.bar(names, values)

# Add graph title and labels
plt.title('Average Gradient Comparison')
plt.xlabel('Activation Functions')
plt.ylabel('Average Absolute Gradient')

# Display grid and graph
plt.grid(axis='y')
plt.show()


# STEP 14: Plot Accuracy Comparison

plt.figure(figsize=(10, 5))

# Plot ReLU validation accuracy
plt.plot(
    relu_history.history['val_sparse_categorical_accuracy'],
    marker='o',
    label='ReLU'
)

# Plot Tanh validation accuracy
plt.plot(
    tanh_history.history['val_sparse_categorical_accuracy'],
    marker='o',
    label='Tanh'
)

# Plot Sigmoid validation accuracy
plt.plot(
    sigmoid_history.history['val_sparse_categorical_accuracy'],
    marker='o',
    label='Sigmoid'
)

# Add title and axis labels
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Display legend, grid and graph
plt.legend()
plt.grid(True)
plt.show()

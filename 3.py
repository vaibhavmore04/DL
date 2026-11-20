
# ASSIGNMENT 3: Experiment with Regularization Techniques

# STEP 1: Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Import neural network model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Import activation functions
from tensorflow.keras.activations import relu, tanh, softmax

# Import L2 regularization
from tensorflow.keras.regularizers import l2


# STEP 2: Load MNIST dataset

# Read training and testing datasets
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

# Separate input pixels and output labels
X_train = train.drop('label', axis=1).values
Y_train = train['label'].values

X_test = test.drop('label', axis=1).values
Y_test = test['label'].values

# Convert images into 28x28 format
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0


# STEP 3: Create Normal Model
# Normal model does not use regularization

model_normal = Sequential([
    Flatten(input_shape=(28, 28)),  # Convert image into 1D

    Dense(256, activation='relu'),  # First hidden layer
    Dense(128, activation='tanh'),  # Second hidden layer
    Dense(64, activation='tanh'),   # Third hidden layer

    Dense(10, activation='softmax') # Output layer
])


# STEP 4: Compile Normal Model

model_normal.compile(
    optimizer='adagrad',                    # Update weights
    loss='sparse_categorical_crossentropy', # Calculate error
    metrics=['accuracy']                    # Measure accuracy
)

# Display model structure
model_normal.summary()


# STEP 5: Train Normal Model

history_normal = model_normal.fit(
    X_train,
    Y_train,
    epochs=10,             # Train for 10 epochs
    validation_split=0.2   # 20% data for validation
)


# STEP 6: Create L2 Regularization Model
# L2 adds a penalty for large weights to reduce overfitting

model_l2 = Sequential([
    Flatten(input_shape=(28, 28)),

    # Apply L2 regularization with value 0.001
    Dense(256, activation='relu',
          kernel_regularizer=l2(0.001)),

    Dense(128, activation='tanh',
          kernel_regularizer=l2(0.001)),

    Dense(64, activation='tanh',
          kernel_regularizer=l2(0.001)),

    Dense(10, activation='softmax')
])


# STEP 7: Compile L2 Model

model_l2.compile(
    optimizer='adagrad',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display L2 model structure
model_l2.summary()


# STEP 8: Train L2 Model

history_l2 = model_l2.fit(
    X_train,
    Y_train,
    epochs=10,
    validation_split=0.2
)


# STEP 9: Create Dropout Model
# Dropout randomly disables some neurons during training
# This helps reduce overfitting

model_dropout = Sequential([
    Flatten(input_shape=(28, 28)),

    Dense(256, activation='relu'),

    # Randomly disable 30% neurons during training
    Dropout(0.3),

    Dense(128, activation='tanh'),

    # Again disable 30% neurons during training
    Dropout(0.3),

    Dense(64, activation='tanh'),

    Dense(10, activation='softmax')
])


# STEP 10: Compile Dropout Model

model_dropout.compile(
    optimizer='adagrad',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display Dropout model structure
model_dropout.summary()


# STEP 11: Train Dropout Model

history_dropout = model_dropout.fit(
    X_train,
    Y_train,
    epochs=10,
    validation_split=0.2
)


# STEP 12: Evaluate All Three Models
# Test models on unseen testing data

# Evaluate Normal Model
normal_loss, normal_accuracy = model_normal.evaluate(
    X_test, Y_test
)

# Evaluate L2 Model
l2_loss, l2_accuracy = model_l2.evaluate(
    X_test, Y_test
)

# Evaluate Dropout Model
dropout_loss, dropout_accuracy = model_dropout.evaluate(
    X_test, Y_test
)


# STEP 13: Plot Accuracy Comparison Graph

# Plot Normal Model accuracy
plt.plot(
    history_normal.history['accuracy'],
    label='Normal Accuracy'
)

# Plot L2 Model accuracy
plt.plot(
    history_l2.history['accuracy'],
    label='L2 Accuracy'
)

# Plot Dropout Model accuracy
plt.plot(
    history_dropout.history['accuracy'],
    label='dropout Accuracy'
)

# Add graph title
plt.title('Accuracy Comparison')

# Display graph legend
plt.legend()

# Plot graph (as in original notebook)
plt.plot()


# STEP 14: Plot Loss Comparison Graph

# Plot Normal Model loss
plt.plot(
    history_normal.history['loss'],
    label='Normal loss'
)

# Plot L2 Model loss
plt.plot(
    history_l2.history['loss'],
    label='L2 loss'
)

# Plot Dropout Model loss
plt.plot(
    history_dropout.history['loss'],
    label='dropout loss'
)

# Add graph title
plt.title('Loss Comparison')

# Display graph legend
plt.legend()

# Plot graph (as in original notebook)
plt.plot()


# STEP 15: Display Final Results

# Normal Model results
print("Normal Model")
print("Test Loss:", normal_loss)
print("Test Accuracy:", normal_accuracy)

# L2 Regularization results
print("\nL2 Model")
print("Test Loss:", l2_loss)
print("Test Accuracy:", l2_accuracy)

# Dropout results
print("\nDropout Model")
print("Test Loss:", dropout_loss)
print("Test Accuracy:", dropout_accuracy)

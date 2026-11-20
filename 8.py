
# ASSIGNMENT 8: Hyperparameter Tuning Using TensorBoard

# STEP 1: Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


# STEP 2: Load MNIST Dataset
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

# Separate images and labels
X_train = train.drop('label', axis=1).values
Y_train = train['label'].values

X_test = test.drop('label', axis=1).values
Y_test = test['label'].values

# Reshape images into 28x28 format
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# STEP 3: Split training data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train,
    Y_train,
    test_size=0.2,
    random_state=42
)


# STEP 4: Create Model 1
# Learning Rate = 0.001, Batch Size = 32, Hidden Layers = 1

model1 = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()

h1 = model1.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[TensorBoard(log_dir='logs/model1')]
)

t1 = time.time() - start


# STEP 5: Create Model 2
# Learning Rate = 0.0001, Batch Size = 32, Hidden Layers = 2

model2 = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()

h2 = model2.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[TensorBoard(log_dir='logs/model2')]
)

t2 = time.time() - start


# STEP 6: Create Model 3
# Learning Rate = 0.001, Batch Size = 64, Hidden Layers = 2

model3 = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()

h3 = model3.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, Y_val),
    callbacks=[TensorBoard(log_dir='logs/model3')]
)

t3 = time.time() - start


# STEP 7: Create Model 4
# Learning Rate = 0.0001, Batch Size = 64, Hidden Layers = 3

model4 = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time.time()

h4 = model4.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, Y_val),
    callbacks=[TensorBoard(log_dir='logs/model4')]
)

t4 = time.time() - start


# STEP 8: Compare Validation Accuracy
plt.figure(figsize=(10, 5))

plt.plot(h1.history['val_accuracy'], label='Model 1')
plt.plot(h2.history['val_accuracy'], label='Model 2')
plt.plot(h3.history['val_accuracy'], label='Model 3')
plt.plot(h4.history['val_accuracy'], label='Model 4')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Hyperparameter Tuning - MNIST')

plt.legend()
plt.grid(True)
plt.show()


# STEP 9: Create Results Comparison Table
results = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],

    'Learning Rate': [0.001, 0.0001, 0.001, 0.0001],

    'Batch Size': [32, 32, 64, 64],

    'Hidden Layers': [1, 2, 2, 3],

    'Validation Accuracy': [
        max(h1.history['val_accuracy']),
        max(h2.history['val_accuracy']),
        max(h3.history['val_accuracy']),
        max(h4.history['val_accuracy'])
    ],

    'Training Time (s)': [t1, t2, t3, t4]
})

print(results)


# STEP 10: Select Best Model
best_index = results['Validation Accuracy'].idxmax()

best_model = [model1, model2, model3, model4][best_index]

# Evaluate on original test dataset
loss, accuracy = best_model.evaluate(X_test, Y_test)

print("Best Model:", results.loc[best_index, 'Model'])
print("Test Accuracy:", accuracy)


# STEP 11: Open TensorBoard
# Run these commands in Jupyter Notebook

%load_ext tensorboard
%tensorboard --logdir logs

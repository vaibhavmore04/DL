
# ASSIGNMENT 7: Transfer Learning Using ResNet50

# STEP 1: Import required libraries
import matplotlib.pyplot as plt    # Plot graphs
import time                        # Measure training time
import tensorflow as tf            # Deep learning library
from tensorflow.keras import layers, Sequential


# STEP 2: Load Cats and Dogs Dataset

# Load training images from folder
train = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\SANCHI\Downloads\DL Practical\cats_and_dogs_filtered\train',
    image_size=(224, 224),
    batch_size=16
)

# Load validation images from folder
val = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\SANCHI\Downloads\DL Practical\cats_and_dogs_filtered\validation',
    image_size=(224, 224),
    batch_size=16
)


# STEP 3: Load Pretrained ResNet50 Model

base = tf.keras.applications.ResNet50(
    weights='imagenet',       # Use pretrained ImageNet weights
    include_top=False,        # Remove original classification layer
    input_shape=(224, 224, 3) # Image dimensions
)

# Freeze pretrained layers
base.trainable = False


# STEP 4: Create Classification Model

model = Sequential([

    # Input layer
    layers.Input(shape=(224, 224, 3)),

    # Preprocess images for ResNet50
    layers.Lambda(tf.keras.applications.resnet50.preprocess_input),

    # Pretrained ResNet50
    base,

    # Convert feature maps into a vector
    layers.GlobalAveragePooling2D(),

    # Hidden layer with 128 neurons
    layers.Dense(128, activation='relu'),

    # Output layer for Cat or Dog
    layers.Dense(1, activation='sigmoid')
])


# STEP 5: Compile Model

model.compile(
    optimizer='adam',            # Update model weights
    loss='binary_crossentropy',  # Loss for binary classification
    metrics=['accuracy']         # Measure accuracy
)


# STEP 6: Train Frozen Model

# Start timer
start = time.time()

# Train for 1 epoch with pretrained layers frozen
h1 = model.fit(
    train,
    epochs=1,
    validation_data=val
)

# Calculate training time
t1 = time.time() - start


# STEP 7: Unfreeze Model for Fine-Tuning

# Allow pretrained layers to be trained
base.trainable = True

# Keep first 10 layers frozen
for layer in base.layers[:10]:
    layer.trainable = False


# STEP 8: Compile Fine-Tuning Model

model.compile(
    # Use smaller learning rate for fine-tuning
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# STEP 9: Train Fine-Tuning Model

# Start timer
start = time.time()

# Train unfrozen model for 1 epoch
h2 = model.fit(
    train,
    epochs=1,
    validation_data=val
)

# Calculate fine-tuning time
t2 = time.time() - start


# STEP 10: Compare Validation Accuracy

# Plot frozen model accuracy
plt.plot(
    h1.history['val_accuracy'],
    label='Frozen Accuracy'
)

# Plot fine-tuned model accuracy
plt.plot(
    h2.history['val_accuracy'],
    label='unfrozen'
)

# Display legend and graph
plt.legend()
plt.show()


# STEP 11: Compare Training Time

# Bar graph for frozen and fine-tuned training time
plt.bar(
    ['Frozen', 'FineTune'],
    [t1, t2]
)

# Display graph
plt.show()


# STEP 12: Save Trained Model

# Save model in Keras format
model.save("cat_dog_model.keras")


# ASSIGNMENT 6(B): Sentiment Analysis Using LSTM

# STEP 1: Import required libraries

import matplotlib.pyplot as plt   # For graphs
import pandas as pd               # For dataset
import numpy as np                # Numerical operations
import tensorflow as tf           # Deep learning library

# Convert text into numbers and pad sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

# Import activation functions
from tensorflow.keras.activations import relu, sigmoid, tanh, softmax

# Split dataset
from sklearn.model_selection import train_test_split


# STEP 2: Load and Preprocess Dataset

# Load IMDB movie reviews
df = pd.read_csv('IMDB Dataset.csv')

# Convert Positive to 1 and Negative to 0
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# Split reviews and labels into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    df['review'],
    df['sentiment'],
    random_state=42,
    test_size=0.2
)


# STEP 3: Tokenization and Padding

# Convert words into numerical values
tokenizer = Tokenizer(
    num_words=10000,
    oov_token='<OOV>'
)

# Learn vocabulary from training data
tokenizer.fit_on_texts(X_train)

# Convert reviews into sequences of length 200
X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=200
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=200
)


# STEP 4: Create LSTM Model

model = Sequential([

    # Convert words into 32-dimensional vectors
    Embedding(10000, 32),

    # LSTM layer with 32 memory units
    LSTM(32),

    # Output layer using sigmoid
    Dense(1, 'sigmoid')
])


# STEP 5: Compile Model

model.compile(
    optimizer='adam',            # Optimizer
    loss='binary_crossentropy',  # Loss function
    metrics=['accuracy']         # Accuracy
)


# STEP 6: Train Model

history = model.fit(
    X_train,
    Y_train,
    epochs=5,              # Train for 5 epochs
    validation_split=0.2   # 20% validation
)


# STEP 7: Evaluate Model

# Evaluate model on testing dataset
loss, accuracy = model.evaluate(X_test, Y_test)


# STEP 8: Plot Accuracy Graph

# Plot training accuracy
plt.plot(
    history.history['accuracy'],
    label='Accuracy'
)

# Plot validation accuracy
plt.plot(
    history.history['val_accuracy'],
    label='Val Accuracy'
)

# Display graph
plt.legend()
plt.show()


# STEP 9: Predict First 5 Reviews

# Predict sentiment of first 5 test reviews
predictions = model.predict(X_test[:5])

for i in range(5):

    # Convert probability into Positive or Negative
    pred = "Positive" if predictions[i][0] >= 0.5 else "Negative"

    # Get actual sentiment from testing labels
    actual = "Positive" if Y_test.iloc[i] == 1 else "Negative"

    # Display prediction and actual result
    print(f"Predicted: {pred}, Actual: {actual}")

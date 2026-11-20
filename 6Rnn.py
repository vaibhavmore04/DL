
# ASSIGNMENT 6(A): Sentiment Analysis Using Simple RNN

# STEP 1: Import required libraries
import matplotlib.pyplot as plt   # For graphs
import pandas as pd               # For dataset
import numpy as np                # For numerical operations
import tensorflow as tf           # Deep learning library

# Convert text into numbers and equal-length sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Split dataset into training and testing
from sklearn.model_selection import train_test_split

# Import neural network model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense


# STEP 2: Load IMDB Dataset

# Read movie review dataset
df = pd.read_csv('IMDB Dataset.csv')


# STEP 3: Preprocess Dataset

# Convert positive to 1 and negative to 0
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# Separate reviews (input) and sentiment (output)
x = df['review'].astype(str)
y = df['sentiment']

# Split 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y,
    random_state=42,
    test_size=0.2
)


# STEP 4: Tokenization and Padding

# Convert words into numbers
# Use maximum 10000 words
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

# Learn words from training reviews
tokenizer.fit_on_texts(X_train)

# Convert reviews into numerical sequences
# Make every review length 200
X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=200
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=200
)


# STEP 5: Create Simple RNN Model

model = Sequential([

    # Convert word numbers into 32-dimensional vectors
    Embedding(10000, 32),

    # RNN layer with 32 units
    SimpleRNN(32),

    # Output layer for Positive or Negative
    Dense(1, activation='sigmoid')
])


# STEP 6: Compile Model

model.compile(
    optimizer='adam',            # Update weights
    loss='binary_crossentropy',  # Loss for 2 classes
    metrics=['accuracy']         # Measure accuracy
)


# STEP 7: Train Model

history = model.fit(
    X_train,
    Y_train,
    epochs=5,              # Train for 5 epochs
    validation_split=0.2   # 20% validation data
)


# STEP 8: Evaluate Model

# Test model on unseen reviews
loss, accuracy = model.evaluate(X_test, Y_test)


# STEP 9: Plot Accuracy Graph

# Training accuracy
plt.plot(
    history.history['accuracy'],
    label='ACCURACY'
)

# Validation accuracy
plt.plot(
    history.history['val_accuracy'],
    label='VALIDATIN ACCURAcY'
)

# Display graph
plt.legend()
plt.show()


# STEP 10: Predict First 5 Reviews

# Predict sentiment for 5 test reviews
predictions = model.predict(X_test[:5])

for i in range(5):

    # Convert prediction into Positive or Negative
    pred = 'positive' if predictions[i][0] >= 0.5 else 'negative'

    # Get actual sentiment
    actual = 'positive' if Y_test.iloc[i] == 1 else 'negative'

    # Display predicted and actual sentiment
    print('Predicted: ', pred, 'Actual: ', actual)

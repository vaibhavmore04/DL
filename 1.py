
# ASSIGNMENT 1: Implementation of Perceptron for AND Gate

# STEP 1: Import Libraries

import numpy as np                 # For mathematical operations
import matplotlib.pyplot as plt    # For plotting graph


# STEP 2: Define Step Activation Function

# Returns 1 if input >= 1, otherwise returns 0
def step_function(x):
    return np.where(x>=1,1,0)


# STEP 3: Create AND Gate Dataset

# Input combinations for AND gate
X=np.array([
    [0,0],[0,1],[1,0],[1,1]
])

# Expected outputs of AND gate
Y=np.array([0,0,0,1])

# Initial weights
weights=np.array([1.2,0.6])

# Initial bias
bias=0.0

# Threshold value
threshold=1.0

# Learning rate controls weight updates
learning_rate=0.5

# Number of training epochs
epochs=10


# STEP 4: Train Perceptron Model

# Repeat training for given epochs
for epoch in range(epochs):

    # Count incorrect predictions
    error=0

    # Process each input combination
    for i in range(len(X)):

        # Calculate weighted sum of inputs
        linear_output=np.dot(X[i],weights)+bias

        # Apply step activation function
        prediction=step_function(linear_output)

        # Calculate difference between target and prediction
        errors=Y[i]-prediction

        # Update weights if prediction is incorrect
        if errors!=0:
            error+=1
            weights+=learning_rate*X[i]*errors

        # Display training results
        print('Input: ',X[i],
              'Target: ',Y[i],
              'Predicted: ',prediction,
              'Weights: ',weights)

    # Stop training when all predictions are correct
    if error==0:
        break

# Display final trained weights
print('Final Weights: ',weights)


# STEP 5: Plot AND Gate Decision Boundary

# Plot input points with colors based on output
plt.scatter(X[:,0],X[:,1],c=Y)

# Define X-axis values for decision boundary
x=np.array([1,0])

# Calculate Y-axis values using weights and threshold
y=(threshold-bias-weights[0]*x)/weights[1]

# Draw decision boundary as dashed black line
plt.plot(x,y,'k--')

# Set graph title
plt.title('AND Gate')

# Display graph
plt.show()

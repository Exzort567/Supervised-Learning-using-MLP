import numpy as np
import pandas as pd
import csv

class MLP:
    def __init__(self, inputSize, noOfHidden, noOfOutput):
        weights_file = 'saved_weights_and_bias.csv'
        try:
            with open(weights_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.hW = np.array(next(reader), dtype=float).reshape((inputSize, noOfHidden))
                self.oW = np.array(next(reader), dtype=float).reshape((noOfHidden, noOfOutput))
                self.hB = np.array(next(reader), dtype=float)
                self.oB = np.array(next(reader), dtype=float)
                self.loaded_weights = True
        except FileNotFoundError:
            # Initialize weights and biases randomly
            self.hW = np.random.randn(inputSize, noOfHidden)
            self.oW = np.random.randn(noOfHidden, noOfOutput)
            self.hB = np.random.randn(noOfHidden)
            self.oB = np.random.randn(noOfOutput)
            self.loaded_weights = False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        hOutput = self.sigmoid(np.dot(x, self.hW) + self.hB)
        yOutput = self.sigmoid(np.dot(hOutput, self.oW) + self.oB)
        return hOutput, yOutput

    def backward(self, y, yPredict, hOutput):
        gradientO = (y - yPredict) * self.derivative_sigmoid(yPredict)
        gradientH = np.dot(gradientO, self.oW.T) * (hOutput * (1 - hOutput))
        return gradientH, gradientO

    def updateHidden(self, learning_rate, gradients, x):
        self.hW += learning_rate * np.outer(x, gradients)
        self.hB += learning_rate * gradients

    def updateOutput(self, learning_rate, gradients, yH):
        self.oW += learning_rate * np.outer(yH, gradients)
        self.oB += learning_rate * gradients
    
    def predict(self, x):
        _, yPredict = self.forward(x)
        # Classify based on the threshold of 0.5
        if yPredict < 0.5:
            return 0
        else:
            return 1
    
    def fit(self, X, Y, learning_rate, max_epoch, threshold):
        if self.loaded_weights:
            print("Weights and biases loaded. Skipping training.")
            return
        
        print("Initializing weights...")
        print("Training model with LR:", learning_rate, "Max Epoch:", max_epoch, "Threshold:", threshold)

        predictions = []

        for epoch in range(1, max_epoch + 1):
            total_loss = 0

            for x, y in zip(X, Y):
                hOutput, yPredict = self.forward(x) 
                gradientH, gradientO = self.backward(y, yPredict, hOutput)
                self.updateHidden(learning_rate, gradientH, x)
                self.updateOutput(learning_rate, gradientO, hOutput)
                total_loss += np.mean((y - yPredict) ** 2)
            
            predictions_epoch = np.array([self.predict(x) for x in X])
            predictions.append(predictions_epoch)

            if epoch % 1 == 0:
                print("Epoch", epoch, "Error:", "{:.4f}".format(total_loss))
                
            if total_loss <= threshold:
                print("Error threshold has been reached. Training stopped.")
                break
            
            elif epoch == max_epoch:
                print("Max epochs reached. Training stopped.")
                
        self.save_weights_and_bias()
        return predictions

    def save_weights_and_bias(self, weightsBias='saved_weights_and_bias.csv'):
        with open(weightsBias, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.hW.flatten())
            writer.writerow(self.oW.flatten())
            writer.writerow(self.hB.flatten())
            writer.writerow(self.oB.flatten())

    def accuracy(self, predictions, labels):
        return (predictions == labels).mean() * 100

# Load dataset
dataset = pd.read_csv('trading_performance.csv')

# Set the random seed for reproducibility
np.random.seed(42)

# Shuffle the dataset
dataset_shuffled = dataset.sample(frac=1).reset_index(drop=True)

# Split dataset into training and testing sets
train_data = dataset_shuffled.sample(frac=0.8)
test_data = dataset_shuffled.drop(train_data.index)

# Extract input features (x1, x2, x3) and target labels (y) for training and testing sets
X_train = train_data[['x1', 'x2', 'x3']].values
Y_train = train_data['y'].values
X_test = test_data[['x1', 'x2', 'x3']].values
Y_test = test_data['y'].values

# Create MLP object
mlp = MLP(inputSize=3, noOfHidden=10, noOfOutput=1)

# Train the MLP
mlp.fit(X_train, Y_train, learning_rate=0.01, max_epoch=100000, threshold=0.01)

# Test predictions
test_predictions = [mlp.predict(x) for x in X_test]
train_predictions = [mlp.predict(x) for x in X_train]

# Print predictions
for x, y_true in zip(X_test, Y_test):
    prediction = mlp.predict(x)
    x_formatted = ', '.join(map(str, x))
    print("Given the X:", x_formatted, "Actual", y_true, "Predicted:", prediction)

# Calculate accuracy on the training set
test_accuracy = mlp.accuracy(test_predictions, Y_test)
print("Test Set Accuracy:", round(test_accuracy, 4), "%")

# Calculate accuracy on the test set
train_accuracy = mlp.accuracy(train_predictions, Y_train)
print("Training Set Accuracy:", round(train_accuracy, 4), "%")
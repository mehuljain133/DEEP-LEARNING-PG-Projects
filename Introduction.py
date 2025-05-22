# Unit-I Introduction: Historical context and motivation for deep learning; basic supervised classification task, optimizing logistic classifier using gradient descent, stochastic gradient descent, momentum, and adaptive sub-gradient method

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_informative=2, n_redundant=0, random_state=42)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
y = y.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function (Binary Cross-Entropy)
def compute_loss(y, y_pred):
    epsilon = 1e-8
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

# Accuracy
def accuracy(y_true, y_pred):
    preds = (y_pred > 0.5).astype(int)
    return np.mean(preds == y_true)

# Training function with different optimizers
def train(X, y, optimizer='gd', lr=0.01, epochs=1000, batch_size=32, beta=0.9):
    m, n = X.shape
    weights = np.zeros((n, 1))
    velocity = np.zeros_like(weights)
    G = np.zeros_like(weights)  # For AdaGrad

    for epoch in range(epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)

        for i in range(0, m, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Forward
            y_pred = sigmoid(X_batch @ weights)
            error = y_pred - y_batch
            grad = X_batch.T @ error / len(y_batch)

            if optimizer == 'gd':
                weights -= lr * grad

            elif optimizer == 'sgd':
                weights -= lr * grad

            elif optimizer == 'momentum':
                velocity = beta * velocity + lr * grad
                weights -= velocity

            elif optimizer == 'adagrad':
                G += grad ** 2
                adjusted_grad = grad / (np.sqrt(G) + 1e-8)
                weights -= lr * adjusted_grad

        if epoch % 100 == 0 or epoch == epochs - 1:
            y_pred_train = sigmoid(X @ weights)
            loss = compute_loss(y, y_pred_train)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return weights

# Run training with different optimizers
for opt in ['gd', 'sgd', 'momentum', 'adagrad']:
    print(f"\n--- Training with {opt.upper()} ---")
    weights = train(X_train, y_train, optimizer=opt, lr=0.1, epochs=1000)
    y_test_pred = sigmoid(X_test @ weights)
    acc = accuracy(y_test, y_test_pred)
    print(f'Test Accuracy: {acc:.4f}')

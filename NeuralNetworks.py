# Unit-II Neural Networks: Feedforward neural networks, deep networks, regularizing a deep network, model exploration, and hyperparameter tuning.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Preprocess
X = StandardScaler().fit_transform(X)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Activation and its derivative
def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)
def softmax(x): 
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Loss and accuracy
def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-8), axis=1))
def accuracy(y, y_hat):
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))

# Neural network class
class NeuralNetwork:
    def __init__(self, layers, lr=0.01, reg=0.0, dropout=0.0):
        self.lr = lr
        self.reg = reg
        self.dropout = dropout
        self.layers = layers
        self.weights = [np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
                        for n_in, n_out in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((1, n)) for n in layers[1:]]

    def forward(self, X, train=True):
        activations = [X]
        dropout_masks = []
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = relu(z)
            if train and self.dropout > 0:
                mask = (np.random.rand(*a.shape) > self.dropout).astype(float)
                a *= mask
                a /= 1 - self.dropout
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
            activations.append(a)
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        a_out = softmax(z_out)
        activations.append(a_out)
        return activations, dropout_masks

    def backward(self, activations, y_true, dropout_masks):
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)
        delta = activations[-1] - y_true
        for i in reversed(range(len(self.weights))):
            grads_w[i] = activations[i].T @ delta + self.reg * self.weights[i]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * d_relu(activations[i])
                if dropout_masks[i - 1] is not None:
                    delta *= dropout_masks[i - 1]
                    delta /= 1 - self.dropout
        return grads_w, grads_b

    def update(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            activations, dropout_masks = self.forward(X, train=True)
            grads_w, grads_b = self.backward(activations, y, dropout_masks)
            self.update(grads_w, grads_b)
            if epoch % 10 == 0 or epoch == epochs - 1:
                loss = cross_entropy(y, activations[-1])
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output, _ = self.forward(X, train=False)
        return output[-1]

# Hyperparameter tuning
param_grid = {
    "layers": [[4, 16, 3], [4, 32, 16, 3]],
    "lr": [0.01, 0.05],
    "reg": [0.0, 0.01],
    "dropout": [0.0, 0.2]
}

best_acc = 0
best_model = None
for params in ParameterGrid(param_grid):
    print(f"\nTraining with params: {params}")
    model = NeuralNetwork(**params)
    model.fit(X_train, y_train, epochs=100)
    preds = model.predict(X_test)
    acc = accuracy(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\nBest Accuracy: {best_acc:.4f}")

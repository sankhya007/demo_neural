import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

class NeuralNetwork:
    def __init__(self, layers, activation='relu', learning_rate=0.01, random_state=42):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.history = {'loss': [], 'accuracy': []}
        
        # Initialize weights and biases
        np.random.seed(self.random_state)
        for i in range(len(layers) - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            weight = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def activation_function(self, x, derivative=False):
        if self.activation == 'sigmoid':
            if derivative:
                return self.activation_function(x) * (1 - self.activation_function(x))
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
        
        elif self.activation == 'relu':
            if derivative:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)
        
        else:
            raise ValueError("Activation function not supported")
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = self.activation_function(z)
            self.z_values.append(z)
            self.activations.append(current_activation)
        
        # Output layer
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        current_activation = self.softmax(z_output)
        self.z_values.append(z_output)
        self.activations.append(current_activation)
        
        return current_activation
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward_propagation(self, X, y):
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = self.activations[-1] - y
        gradients_w[-1] = np.dot(self.activations[-2].T, delta) / m
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers gradients
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_function(self.z_values[l], derivative=True)
            gradients_w[l] = np.dot(self.activations[l].T, delta) / m
            gradients_b[l] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def fit(self, X, y, epochs=1000, batch_size=32, validation_data=None, verbose=True):
        X_train, y_train = X, y
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                y_pred = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch)
            
            # Calculate metrics
            y_pred_full = self.forward_propagation(X_train)
            train_loss = self.compute_loss(y_train, y_pred_full)
            train_accuracy = self.accuracy(y_train, y_pred_full)
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_accuracy)
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward_propagation(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                val_accuracy = self.accuracy(y_val, y_val_pred)
            
            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}"
                if validation_data is not None:
                    msg += f", Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
                print(msg)
    
    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, y_true, y_pred):
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
            
        y_pred_labels = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(self.history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
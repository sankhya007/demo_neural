import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='relu', learning_rate=0.01, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if activation == 'relu':
                std = np.sqrt(2.0 / layers[i])
            else:
                std = np.sqrt(1.0 / layers[i])
                
            weight_matrix = np.random.randn(layers[i], layers[i + 1]) * std
            bias_vector = np.zeros((1, layers[i + 1]))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
    
    def activation_function(self, x, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            if derivative:
                sig = 1 / (1 + np.exp(-x))
                return sig * (1 - sig)
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
    
    def softmax(self, x):
        # Numerical stability improvement
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.activation_function(z)
            self.activations.append(current_activation)
        
        # Output layer (softmax)
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output_activation = self.softmax(z_output)
        self.activations.append(output_activation)
        
        return output_activation
    
    def backward_propagation(self, X, y):
        m = X.shape[0]
        
        # Calculate output layer error
        delta = self.activations[-1] - y
        
        # Update weights and biases for output layer
        self.weights[-1] -= self.learning_rate * np.dot(self.activations[-2].T, delta) / m
        self.biases[-1] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_function(self.z_values[i], derivative=True)
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, delta) / m
            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
    
    def compute_loss(self, y_true, y_pred):
        # Categorical cross-entropy loss
        epsilon = 1e-15  # For numerical stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def accuracy(self, y_true, y_pred):
        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predicted_labels == true_labels)
    
    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate_in_batches(self, X, y, batch_size=1000):
        """Evaluate model on large datasets in batches to avoid memory issues"""
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            y_pred = self.forward_propagation(X_batch)
            batch_loss = self.compute_loss(y_batch, y_pred)
            batch_accuracy = self.accuracy(y_batch, y_pred)
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            num_batches += 1
        
        return total_loss / num_batches, total_accuracy / num_batches
from neural_network import NeuralNetwork
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def test_classification():
    """Test the neural network on a classification problem."""
    print("Testing Neural Network on Classification Problem...")
    
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=15, random_state=42)
    
    # Convert to one-hot encoding
    y_one_hot = np.eye(3)[y]
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train neural network
    nn = NeuralNetwork(layers=[20, 64, 32, 3], activation='relu', learning_rate=0.01)
    nn.fit(X_train, y_train, epochs=1000, batch_size=32, 
           validation_data=(X_test, y_test), verbose=True)
    
    # Evaluate - FIXED THIS LINE
    y_pred = nn.predict(X_test)
    test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)  # FIXED: removed extra parameter
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    nn.plot_training_history()
    
    return nn

def test_regression():
    """Test the neural network on a regression problem."""
    print("\nTesting Neural Network on Regression Problem...")
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=1000, n_features=10, n_targets=1, noise=0.1, random_state=42)
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))
    
    # For regression, we need a completely separate class
    class NeuralNetworkRegressor:
        def __init__(self, layers, activation='relu', learning_rate=0.01, random_state=42):
            self.layers = layers
            self.activation = activation
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.weights = []
            self.biases = []
            self.history = {'loss': []}
            
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
            
            # Output layer (linear activation for regression)
            z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
            current_activation = z_output  # Linear activation
            self.z_values.append(z_output)
            self.activations.append(current_activation)
            
            return current_activation
        
        def compute_loss(self, y_true, y_pred):
            """Compute mean squared error for regression."""
            return mean_squared_error(y_true, y_pred)
        
        def backward_propagation(self, X, y):
            m = X.shape[0]
            gradients_w = [np.zeros_like(w) for w in self.weights]
            gradients_b = [np.zeros_like(b) for b in self.biases]
            
            # Output layer gradient (linear activation derivative is 1)
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
                
                self.history['loss'].append(train_loss)
                
                # Validation metrics
                val_loss = None
                if validation_data is not None:
                    X_val, y_val = validation_data
                    y_val_pred = self.forward_propagation(X_val)
                    val_loss = self.compute_loss(y_val, y_val_pred)
                
                if verbose and epoch % 100 == 0:
                    msg = f"Epoch {epoch}: Loss = {train_loss:.4f}"
                    if val_loss is not None:
                        msg += f", Val Loss = {val_loss:.4f}"
                    print(msg)
        
        def predict(self, X):
            return self.forward_propagation(X)
        
        def plot_training_history(self):
            plt.figure(figsize=(10, 4))
            plt.plot(self.history['loss'])
            plt.title('Training Loss (MSE)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.show()
    
    # Create and train neural network for regression
    nn_reg = NeuralNetworkRegressor(layers=[10, 64, 32, 1], activation='relu', learning_rate=0.01)
    
    print("Training regression network...")
    nn_reg.fit(X_train, y_train, epochs=1000, batch_size=32, 
               validation_data=(X_test, y_test), verbose=True)
    
    # Evaluate
    y_pred = nn_reg.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"Final Test MSE: {test_mse:.4f}")
    print(f"Final Test R² Score: {test_r2:.4f}")
    
    # Plot some results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn_reg.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {test_r2:.4f})')
    
    plt.tight_layout()
    plt.show()
    
    return nn_reg

if __name__ == "__main__":
    # Test the neural network
    print("Starting Neural Network Demo...")
    nn_classifier = test_classification()
    nn_regressor = test_regression()
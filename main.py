import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
import time
import sys
import itertools
from sklearn.metrics import classification_report, confusion_matrix

# LinkedIn-style header with animations
def print_header():
    print("\n" + "🚀" + "="*70 + "🚀")
    print("           🤖 NEURAL NETWORK - MNIST DIGIT RECOGNITION")
    print("                 Professional LinkedIn Demo")
    print("🚀" + "="*70 + "🚀")
    print()

def animated_loading(text, duration=2):
    """Show animated loading spinner"""
    spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
    end_time = time.time() + duration
    while time.time() < end_time:
        sys.stdout.write(f'\r{next(spinner)} {text}...')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\r✅ {text} completed!{" " * 50}\n')

def progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Display progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

# Enhanced training output
class TrainingMonitor:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.emoji_cycle = itertools.cycle(['🧠', '⚡', '🚀', '🔥', '💫', '🌟'])
        
    def update(self, epoch, train_loss, train_accuracy, val_loss=None, val_accuracy=None):
        elapsed = time.time() - self.start_time
        emoji = next(self.emoji_cycle)
        
        # Progress bar
        progress_bar(epoch + 1, self.total_epochs, 
                    prefix=f'{emoji} Training', 
                    suffix=f'Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | Time: {elapsed:.1f}s')
        
        # Detailed output every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n   📊 Epoch {epoch + 1:3d}: ", end="")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}", end="")
            if val_loss is not None:
                print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            else:
                print()

print_header()

# Load MNIST dataset with animation
print("📥 DATASET LOADING")
print("-" * 50)
animated_loading("Downloading MNIST dataset", 1)

# Use a smaller subset for memory efficiency
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

animated_loading("Preprocessing data", 1)

# Normalize pixel values from 0-255 to 0-1 and convert to float32 for memory efficiency
X = (X / 255.0).astype(np.float32)

# Convert labels to one-hot encoding
def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_one_hot = to_one_hot(y)

# Use smaller subsets for memory efficiency - FURTHER REDUCED
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_one_hot, test_size=2000, random_state=42  # Reduced test size
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2000, random_state=42  # Reduced validation size
)

# Clear temporary variables to free memory
del X_temp, y_temp, X, y_one_hot
import gc
gc.collect()

print("\n📊 DATASET OVERVIEW")
print("-" * 50)
print(f"🎯 Training samples:   {X_train.shape[0]:,}")
print(f"📈 Validation samples: {X_val.shape[0]:,}")
print(f"🧪 Test samples:      {X_test.shape[0]:,}")
print(f"🖼️  Image dimensions:  28×28 pixels")
print(f"🔢 Number of classes:  10 (digits 0-9)")

# Create and train the neural network with smaller architecture
print("\n🔧 NEURAL NETWORK ARCHITECTURE")
print("-" * 50)
nn = NeuralNetwork(
    layers=[784, 32, 16, 10],  # Even smaller architecture
    activation='relu',
    learning_rate=0.01,
    random_state=42
)

print(f"🏗️  Network Layers:    {nn.layers}")
print(f"⚡ Activation:         {nn.activation}")
print(f"📚 Learning Rate:      {nn.learning_rate}")
print(f"🎲 Random Seed:        {nn.random_state}")

# Enhanced training with monitor
print("\n🚀 MODEL TRAINING")
print("-" * 50)
print("🎯 Starting neural network training...\n")

# Train with smaller batch size and fewer epochs for memory efficiency
start_time = time.time()
monitor = TrainingMonitor(20)  # Further reduced epochs

for epoch in range(20):
    # Memory-efficient mini-batch training
    m = X_train.shape[0]
    indices = np.random.permutation(m)
    batch_size = 32  # Smaller batch size
    
    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        y_pred = nn.forward_propagation(X_batch)
        nn.backward_propagation(X_batch, y_batch)
    
    # Calculate metrics using batch evaluation to avoid memory issues
    train_loss, train_accuracy = nn.evaluate_in_batches(X_train, y_train, batch_size=1000)
    val_loss, val_accuracy = nn.evaluate_in_batches(X_val, y_val, batch_size=1000)
    
    nn.history['loss'].append(train_loss)
    nn.history['accuracy'].append(train_accuracy)
    
    # Update monitor
    monitor.update(epoch, train_loss, train_accuracy, val_loss, val_accuracy)

training_time = time.time() - start_time
print(f"\n✅ Training completed!{' ' * 60}")

# Evaluate on test set in batches
print("\n📊 MODEL EVALUATION")
print("-" * 50)
animated_loading("Running final evaluation", 1)

test_loss, test_accuracy = nn.evaluate_in_batches(X_test, y_test, batch_size=1000)

# Performance rating
if test_accuracy > 0.97:
    performance = "🎉 EXCELLENT"
    color = '#2E8B57'  # Green
elif test_accuracy > 0.95:
    performance = "🔥 OUTSTANDING" 
    color = '#228B22'  # Forest Green
elif test_accuracy > 0.92:
    performance = "⚡ GREAT"
    color = '#32CD32'  # Lime Green
elif test_accuracy > 0.90:
    performance = "🚀 GOOD"
    color = '#FFD700'  # Gold
else:
    performance = "📈 DECENT"
    color = '#FF8C00'  # Dark Orange

print(f"\n🏆 FINAL RESULTS")
print("-" * 30)
print(f"✅ Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📉 Test Loss:      {test_loss:.4f}")
print(f"⏱️  Training Time:  {training_time:.2f} seconds")
print(f"🎯 Performance:     {performance}")

# Enhanced LinkedIn-ready plots
print("\n📈 GENERATING PROFESSIONAL VISUALIZATIONS")
print("-" * 55)

plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Neural Network Performance - MNIST Digit Recognition\nLinkedIn Ready Demo', 
             fontsize=24, fontweight='bold', y=0.95)

# Plot 1: Training History
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax1.plot(nn.history['loss'], linewidth=3, color='#FF6B6B', alpha=0.8, label='Training Loss')
ax1.set_title('📉 Training Progress - Loss & Accuracy', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=14, fontweight='bold', color='#FF6B6B')
ax1.tick_params(axis='y', labelcolor='#FF6B6B')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=12)

ax1_twin = ax1.twinx()
ax1_twin.plot(nn.history['accuracy'], linewidth=3, color='#4ECDC4', alpha=0.8, label='Training Accuracy')
ax1_twin.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='#4ECDC4')
ax1_twin.tick_params(axis='y', labelcolor='#4ECDC4')
ax1_twin.legend(loc='upper right', fontsize=12)

# Plot 2: Performance Summary
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax2.text(0.5, 0.7, f'{test_accuracy*100:.2f}%', fontsize=48, ha='center', va='center', 
         fontweight='bold', color=color)
ax2.text(0.5, 0.5, 'Test Accuracy', fontsize=18, ha='center', va='center', fontweight='bold')
ax2.text(0.5, 0.3, f'{X_test.shape[0]:,} samples', fontsize=14, ha='center', va='center', alpha=0.7)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('🎯 Final Performance', fontsize=16, fontweight='bold', pad=20)

# Plot 3: Architecture Diagram
ax3 = plt.subplot2grid((3, 2), (1, 1))
layers = nn.layers
for i, (layer, size) in enumerate(zip(['Input', 'Hidden 1', 'Hidden 2', 'Output'], layers)):
    ax3.text(0.1, 0.9 - i*0.2, f'{layer}: {size} neurons', fontsize=14, 
             fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.1, f'Total Parameters: ~{sum(layers[i]*layers[i+1] for i in range(len(layers)-1)):,}', 
         fontsize=12, ha='center', transform=ax3.transAxes, style='italic')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('🏗️ Network Architecture', fontsize=16, fontweight='bold', pad=20)

# Plot 4: Sample Predictions
ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
def enhanced_visualize_predictions(ax, X, y_true, model, num_samples=12):
    indices = np.random.choice(len(X), num_samples)
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        image = X[idx].reshape(28, 28)
        prediction = model.predict(X[idx:idx+1])[0]
        true_label = np.argmax(y_true[idx])
        
        row = i // 6
        col = i % 6
        
        ax.imshow(image, extent=[col, col+1, 2-row, 3-row], cmap='Blues', alpha=0.8)
        
        if true_label == prediction:
            correct_predictions += 1
            color = '#2E8B57'
            marker = '✅'
        else:
            color = '#DC143C'
            marker = '❌'
        
        ax.text(col + 0.5, 2.8 - row, f'{marker}\nT:{true_label} P:{prediction}', 
                fontsize=10, ha='center', va='center', color=color, fontweight='bold')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'🔍 Sample Predictions ({correct_predictions}/{num_samples} Correct)', 
                 fontsize=16, fontweight='bold', pad=20)

enhanced_visualize_predictions(ax4, X_test, y_test, nn)

plt.tight_layout()
plt.savefig('linkedin_mnist_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("💾 Professional visualization saved as 'linkedin_mnist_demo.png'")

# Enhanced classification report (using batches)
print("\n📝 DETAILED PERFORMANCE ANALYSIS")
print("-" * 45)

# Predict in batches for the classification report
y_test_labels = []
y_pred_labels = []

batch_size = 1000
for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i + batch_size]
    y_batch = y_test[i:i + batch_size]
    
    y_test_labels.extend(np.argmax(y_batch, axis=1))
    y_pred_labels.extend(nn.predict(X_batch))

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, digits=4))

# Final LinkedIn summary
print("\n" + "🎊" + "="*70 + "🎊")
print("              🏆 TRAINING COMPLETED SUCCESSFULLY!")
print(f"           Final Accuracy: {test_accuracy*100:.2f}% on MNIST Dataset")
print("              Ready for LinkedIn Sharing! 🚀")
print("🎊" + "="*70 + "🎊")

print("\n✨ POST THIS ON LINKEDIN:")
print("   📸 Share the generated 'linkedin_mnist_demo.png' image")
print("   📊 Mention your final accuracy score")
print("   🤖 Talk about building a neural network from scratch")
print("   💡 Highlight the architecture and training process")
print("   🔗 Use hashtags: #AI #MachineLearning #NeuralNetworks #Python #DeepLearning")

# Save the model
def save_model(model, filename):
    model_data = {
        'weights': [w.astype(np.float32) for w in model.weights],  # Save as float32 for efficiency
        'biases': [b.astype(np.float32) for b in model.biases],
        'layers': model.layers,
        'activation': model.activation
    }
    np.save(filename, model_data)
    print(f"\n💾 Model saved to '{filename}'")

save_model(nn, 'mnist_model.npy')
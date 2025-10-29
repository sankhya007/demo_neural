```markdown
# 🧠 Neural Network from Scratch (Pure Python Implementation)

A fully functional **neural network built from scratch in Python**, without relying on deep learning frameworks like TensorFlow or PyTorch.  
This project demonstrates the **core mechanics** behind neural networks — from forward propagation to backpropagation and gradient descent — with clean, modular, and well-documented code.

---

## 🚀 Features

✅ **Custom Neural Network Class** – Pure Python + NumPy implementation  
✅ **Multiple Activation Functions** – ReLU, Sigmoid, and Tanh  
✅ **Configurable Architecture** – Define any number of layers and neurons  
✅ **Supports Both Tasks** – Classification & Regression  
✅ **Mini-Batch Gradient Descent** – Efficient training  
✅ **Real-Time Training Visualization** – Loss and accuracy plots  
✅ **Comprehensive Evaluation Metrics** – Accuracy, MSE, and R² score  

---

## 🧩 Project Structure

```
neural_network_demo/
├── neural_network.py      # Core neural network implementation (classification)
├── main.py                # Demo for classification and regression
├── requirements.txt       # Dependencies
├── run.bat                # Run script for Windows
├── run.sh                 # Run script for Linux/Mac
├── README.md              # Project documentation
└── .gitignore             # Ignore unnecessary files
```

---

## ⚙️ Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/sankhya007/neural_network_demo.git
cd neural_network_demo
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the demo:
```bash
python main.py
```

You’ll see:
- Training progress with live updates every 100 epochs  
- Real-time loss and accuracy visualization  
- Final evaluation metrics for both classification and regression tasks  

---

## 📊 Results

- Achieves **~78% accuracy** on multi-class classification datasets  
- Smooth convergence curves for both **loss** and **accuracy**
- Regression model yields strong **R² performance** and low **MSE**  

---

## 🧠 Concepts Covered

- Forward Propagation  
- Backpropagation  
- Activation Functions (ReLU, Sigmoid, Tanh)  
- Softmax Output for Multi-Class Problems  
- Mini-Batch Gradient Descent  
- Loss Functions (Cross-Entropy & MSE)  
- Model Evaluation Metrics  

---

## 📈 Example Visualizations

- **Classification:** Loss & Accuracy curves over epochs  
- **Regression:** Predicted vs Actual scatter plot with R² score  
- **Training Diagnostics:** Validation loss tracking

---

## 💡 Future Improvements

- Add dropout and batch normalization  
- Implement momentum and Adam optimizers  
- Save and load trained model weights  
- Extend to support convolutional layers

---

## 🧑‍💻 Author

**Sankhyapriyo Dey**  
📧 [GitHub Profile](https://github.com/sankhya007)  

---

## 🪪 License

This project is released under the **MIT License** – free to use and modify.

---
```
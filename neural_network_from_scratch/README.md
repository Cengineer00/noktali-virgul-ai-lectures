# Neural Network from Scratch

## 📚 Course Overview

This lesson demonstrates how to build a **Neural Network** from scratch using only Python and NumPy, without any deep learning frameworks like PyTorch, TensorFlow, or Keras. The implementation includes forward propagation, backpropagation, and training on MNIST and CIFAR10 datasets.

### Learning Objectives
- Understand the fundamentals of neural network architecture
- Learn forward and backward propagation algorithms
- Implement activation functions (ReLU, Softmax)
- Master gradient descent optimization
- Apply the network to real-world image classification tasks

---

## 🎯 Prerequisites

### Required Libraries
```bash
pip install numpy torchvision jupyter
```
**Note:** `torchvision` is only used for downloading CIFAR10 dataset, not for neural network implementation.

### Knowledge Requirements
- Solid Python programming skills
- Strong understanding of NumPy operations
- Linear algebra basics (matrix multiplication, dot products)
- Calculus fundamentals (derivatives, chain rule)
- Basic understanding of machine learning concepts

---

## 🔍 Code Structure

### Core Components

#### 1. **LinearLayer Class**
Implements a fully connected (dense) layer:
- **Initialization**: Uses He initialization for weights (`W ~ N(0, sqrt(2/n_in))`)
- **`forward(input_data)`**: Computes `output = input × W + b`
- **`backward(output_gradient, learning_rate)`**: Computes gradients and updates weights/biases

**Mathematical Operations:**
```
Forward:  Z = X × W + b
Backward: ∂W = X^T × ∂Z
          ∂X = ∂Z × W^T
```

#### 2. **ReLULayer Class**
Rectified Linear Unit activation:
- **`forward(input_data)`**: Returns `max(0, input_data)`
- **`backward(output_gradient, learning_rate)`**: Passes gradient only for positive inputs

**Formula:**
```
ReLU(x) = max(0, x)
∂ReLU/∂x = 1 if x > 0, else 0
```

#### 3. **SoftmaxLayer Class**
Converts logits to probability distribution:
- **`forward(input_data)`**: Applies softmax normalization
- **`backward(output_gradient, learning_rate)`**: Passes gradient through (combined with cross-entropy)

**Formula:**
```
Softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

#### 4. **CrossEntropyLoss Class**
Computes classification loss:
- **`forward(predictions, targets)`**: Calculates negative log-likelihood
- **`backward()`**: Computes gradient: `(predictions - one_hot_targets) / batch_size`

**Formula:**
```
Loss = -1/m × Σ log(p_correct_class)
```

#### 5. **NeuralNetwork Class**
Main network architecture:
- **Structure**: Input → Linear → ReLU → Linear → ReLU → Softmax
- **`forward(x)`**: Performs forward pass through all layers
- **`backward(loss_grad, learning_rate)`**: Backpropagates gradients and updates weights

---

## 🚀 Usage Example

### Training on MNIST

```python
from neural_network import NeuralNetwork
import numpy as np

# Load MNIST data (28x28 grayscale images, 10 classes)
X_train = X_train.reshape(-1, 784) / 255.0  # Flatten and normalize
Y_train = Y_train.astype(int)

# Initialize network
model = NeuralNetwork(
    input_size=784,   # 28x28 pixels
    hidden_size=128,  # Hidden layer neurons
    output_size=10    # 10 digit classes
)

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model.forward(X_batch)
    
    # Compute loss
    loss = model.loss_function.forward(predictions, Y_batch)
    
    # Backward pass
    loss_gradient = model.loss_function.backward()
    model.backward(loss_gradient, learning_rate)
```

### Training on CIFAR10

Similar approach but with:
- Input size: 3072 (32x32x3 RGB images)
- More complex architecture recommended
- Data augmentation for better performance

---

## 💡 Key Concepts

### Forward Propagation
Data flows through layers sequentially:
```
Input → Linear1 → ReLU → Linear2 → ReLU → Softmax → Output
```

### Backpropagation
Gradients flow backward using chain rule:
```
Loss ← Softmax ← ReLU ← Linear2 ← ReLU ← Linear1 ← Input
```

### Weight Initialization
- **He Initialization**: Prevents vanishing/exploding gradients in ReLU networks
- Formula: `W ~ N(0, sqrt(2 / n_input))`

### Gradient Descent
- Updates weights in the direction that reduces loss
- Formula: `W_new = W_old - learning_rate × ∂Loss/∂W`

---

## 📊 Performance Tips

1. **Normalization**: Scale input data to [0, 1] range
2. **Batch Processing**: Process multiple samples simultaneously
3. **Learning Rate**: Start with 0.001-0.01, adjust based on convergence
4. **Hidden Layer Size**: Balance between capacity and overfitting
5. **Epochs**: Monitor validation loss to prevent overfitting

---

## 🎓 Key Takeaways

1. **Layer Abstraction**: Each layer handles its own forward/backward operations
2. **Automatic Differentiation**: Backpropagation computes gradients automatically
3. **Modular Design**: Easy to add new layers or activation functions
4. **Pure NumPy**: Understanding low-level operations builds strong foundations
5. **Scalability**: Same principles apply to deeper, more complex networks

---

## 📝 Additional Files

- **`neural_network_mnist.ipynb`**: Complete MNIST training notebook
- **`neural_network_cifar10.ipynb`**: CIFAR10 training with visualizations
- **`readme.md`**: Original Turkish documentation

---

## 🔗 Resources

For a detailed video explanation of this implementation, check out:
[Sıfırdan Yapay Sinir Ağı Geliştirdim](https://youtu.be/witsTpml9YM?si=ZXiH-ehFT5fVM1uH)

---

## ⚡ Next Steps

- Experiment with different architectures (more layers, different sizes)
- Implement other activation functions (Sigmoid, Tanh, Leaky ReLU)
- Add regularization techniques (L2, Dropout)
- Try different optimizers (Momentum, Adam)
- Apply to other datasets

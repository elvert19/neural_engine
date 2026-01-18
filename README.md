

---

# Neural Engine

**Neural Engine** is a high-performance deep learning library built with a **Rust** backend for speed and safety, and exposed to **Python** for ease of use. It serves as a lightweight, "micro-PyTorch" that allows users to build, train, and deploy neural networks efficiently.

## Project Structure

Here is a breakdown of the core files in `src/` and what they do:

### 1. `src/lib.rs` (The Bridge)

This is the entry point for the **PyO3** bindings. It acts as the interface between Python and Rust.

* Defines the `Sequential` class exposed to Python.
* Handles type conversion between Python lists/Numpy arrays and Rust `ndarray`.
* Orchestrates the training loop, delegating tasks to the other modules.

### 2. `src/layer.rs` (The Architecture)

Contains the implementation of neural network layers.

* **`Dense` Layer:** Implements a fully connected layer. It manages weights and biases, computes the forward pass (linear transformation), and calculates gradients during backpropagation.

### 3. `src/activation.rs` (The Non-Linearity)

Defines activation functions that introduce non-linearity to the network.

* **ReLU:** Rectified Linear Unit, used for hidden layers.
* **Sigmoid:** Squashes output between 0 and 1, used for binary classification.
* **Tanh:** Squashes output between -1 and 1, often used in hidden layers.

### 4. `src/loss.rs` (The Error Calculation)

Contains the mathematical functions to measure how well the model is performing. (See "Mathematical Foundations" below).

### 5. `src/optimizer.rs` (The Learning)

Implements algorithms that update the weights based on gradients.

* **SGD:** Stochastic Gradient Descent.
* **Adam:** Adaptive Moment Estimation (momentum-based updates).

### 6. `src/utilis.rs` & `src/tensor.rs`

Helper modules for matrix operations and type definitions (aliasing `ndarray::Array2<f64>` as `Tensor`).

---

## Mathematical Foundations

The core logic of the engine relies on the following mathematical principles.

### 1. Loss Functions

#### Mean Squared Error (MSE)
Used primarily for **Regression** problems.
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$




**Derivative for Backpropagation:**
$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n} (\hat{y} - y)$$



#### Binary Cross-Entropy (BCE)
Used for **Binary Classification**.
$$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i)]$$



### 2. Activation Functions

* **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
 
* **Tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
 
* **ReLU:** $f(x) = \max(0, x)$
  

---

## Installation & Usage

### Prerequisites

* Rust (Cargo)
* Python 3.x
* `maturin` (`pip install maturin`)

### Build & Install

Run this in the project root to compile the Rust backend and install it into your Python environment since we cannot be able to use cargo functions in this case:

```bash
maturin develop --release

```

### Python Example

```python
import neural_engine
import numpy as np

# Define Model
model = neural_engine.Sequential()
model.add_dense(2, 4)       # Input: 2, Hidden: 4
model.add_tanh()            # Activation
model.add_dense(4, 1)       # Hidden: 4, Output: 1
model.add_sigmoid()         # Output activation

# Configure
model.set_optimizer("adam", 0.01)
model.set_loss("mse")

# Train
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
model.train(X, y, epochs=1000)

```

---

## Performance Results (XOR Problem)

The following output demonstrates the engine solving the non-linear XOR problem. Note the rapid convergence around Epoch 400.

```text
--- Setting up XOR Test ---

--- Starting Training ---
Epoch 0: Loss = 0.2603
Epoch 100: Loss = 0.1255
Epoch 200: Loss = 0.1252
Epoch 300: Loss = 0.1251
Epoch 400: Loss = 0.0002   <-- Network converges here
Epoch 500: Loss = 0.0000
Epoch 600: Loss = 0.0000
Epoch 700: Loss = 0.0000
Epoch 800: Loss = 0.0000
Epoch 900: Loss = 0.0000

Final Loss: 0.000006

--- Predictions ---
Input: [0. 0.] | Target: [0.] | Prediction: 0.0008
Input: [0. 1.] | Target: [1.] | Prediction: 0.9970
Input: [1. 0.] | Target: [1.] | Prediction: 0.9974
Input: [1. 1.] | Target: [0.] | Prediction: 0.0027

```

---

## Future Implementations

The goal is to expand `neural_engine` into a robust tool for larger datasets and more complex architectures.

1. **Model Serialization (Save/Load):**
* Implement functionality to save trained weights to a `.json` or binary file so models can be reused without retraining.


2. **Summary Method:**
* Add a `model.summary()` method to print the network topology (layer shapes and parameter counts).


3. **Convolutional Layers (Conv2D):**
* Implement 2D convolution to handle Image Processing tasks (like MNIST or CIFAR-10).


4. **Batch Processing:**
* Currently, the engine processes full-batch updates. Implementing "Mini-batch" gradient descent will allow training on massive datasets that don't fit in memory.


5. **Softmax & Categorical Cross-Entropy:**
* Support for multi-class classification (e.g., predicting digits 0-9).

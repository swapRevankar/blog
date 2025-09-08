---
title: "From Math to Code: Logistic Regression in Neural Networks"
date: 2025-09-07
series: ["Deep Learning Specialization"]
series_weight: 1
tags: ["coursera", "deeplearning"]
math: true
---

I’m currently taking **Neural Networks and Deep Learning** (part of the [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)).  
One of the first building blocks we meet is **logistic regression**.  

This post is part of my learning journal — my goal is to document the flow from **math → code** instead of re-explaining the theory.

---

## Preparing the Data

Before training, images must be put into a format the algorithm understands.

### 1) Dataset dimensions

When loaded, each image has 3 dimensions: height, width, and RGB channels.  

- `n_train` → number of training images  
- `n_test` → number of test images  
- Each image: `img_size × img_size × 3`  

```python
n_train  = X_train.shape[0]
n_test   = X_test.shape[0]
img_size = X_train.shape[1]
```

Example: for 64×64 RGB images, each picture has **64 × 64 × 3 = 12,288 numbers**.

---

### 2) Flattening

Neural nets expect **vectors, not cubes**.  
So we “unroll” each image into a column vector, with one column per image.

```python
X_train_flat = X_train.reshape(n_train, -1).T
X_test_flat  = X_test.reshape(n_test, -1).T
```

- Before: `(n_train, 64, 64, 3)`  
- After: `(12288, n_train)`  

Think of taking a Rubik’s cube and stretching it into a line of numbers.

---

### 3) Normalization

Pixel values range from `0 → 255`. Scaling them to `[0, 1]` helps learning.

```python
X_train = X_train_flat / 255.
X_test  = X_test_flat  / 255.
```

At this point:  
- Training set → `(12288, n_train)`  
- Test set → `(12288, n_test)`  
- Values between 0 and 1.

---

## Building Blocks of Logistic Regression

Now we construct the parts of the algorithm step by step.

---

### Parameters `(w, b)`

- `w` = weights (how important each pixel is)  
- `b` = bias (a constant offset)

```python
import numpy as np

def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b
```

Here, `dim = number of features` (e.g., 12,288 for 64×64×3).

---

### Activation: Sigmoid

**Math**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^\top X + b
$$

**Code**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

### Cost and Gradients

We measure how far predictions are from labels and compute gradients for updates.

**Math**

- Prediction: \( A = \sigma(w^\top X + b) \)  
- Cost:
  $$
  J = -\frac{1}{m} \sum_{i=1}^m \Big[y^{(i)} \log A^{(i)} + (1-y^{(i)}) \log(1-A^{(i)})\Big]
  $$
- Gradients:
  $$
  dw = \frac{1}{m} X (A - Y)^\top,\quad db = \frac{1}{m} \sum_{i=1}^m (A^{(i)} - y^{(i)})
  $$

**Code**

```python
def propagate(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    # cost
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1-A))

    # gradients
    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)

    grads = {"dw": dw, "db": db}
    return grads, cost
```

---

### Gradient Descent

We repeatedly update `w` and `b`.

**Math**

$$
w := w - \alpha \, dw \newline
b := b - \alpha \, db
$$

**Code**

```python
def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.5, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")

    params = {"w": w, "b": b}
    return params, grads, costs
```

---

### Prediction

Turn probabilities into binary outputs.

**Math**

$$
\hat{y}^{(i)} =
\begin{cases}
1 & \text{if } A^{(i)} > 0.5 \newline
0 & \text{otherwise}
\end{cases}
$$

**Code**

```python
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)
```

---

## The Full Model

Now we merge everything into a single function.

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    n_x = X_train.shape[0]
    w, b = initialize(n_x)

    params, grads, costs = optimize(
        w, b, X_train, Y_train,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        print_cost=print_cost
    )
    w, b = params["w"], params["b"]

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test  = predict(w, b, X_test)

    train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
    test_acc  = 100 - np.mean(np.abs(Y_pred_test  - Y_test)) * 100

    return {
        "costs": costs,
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test":  Y_pred_test
    }
```

---

## Visualizing Training (Optional)

```python
import matplotlib.pyplot as plt

def plot_cost(costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (x100)")
    plt.title("Learning curve")
    plt.show()
```

---

## Wrap-up

We’ve gone from raw images to a working logistic regression classifier:

1. **Preprocessing**: flatten + normalize  
2. **Initialize**: weights and bias  
3. **Forward + Cost + Backward**: compute activations and gradients  
4. **Optimize**: gradient descent  
5. **Predict**: binary classification  

This forms the foundation for neural networks, where logistic regression units are stacked into layers.

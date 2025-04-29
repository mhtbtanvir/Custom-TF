# ATRT: Attention-Regularized Transformer for Text Classification

A lightweight Transformer model built from scratch in TensorFlow with a custom attention mechanism designed to improve convergence and training stability. The model was tested on the IMDb movie reviews dataset for binary sentiment classification.

## ✨ Highlights

- ✅ **Custom Transformer Architecture** built from scratch using TensorFlow
- 🧠 **Gaussian-Scaled Multi-Head Attention** for improved focus and early convergence
- 📈 Achieved **78% test accuracy** on the IMDb dataset
- ⚙️ Implemented **dynamic positional encoding**, dropout, and regularization
- 🧪 Modular and readable code using **TensorFlow Datasets** and `TextVectorization`

## 🗂️ Dataset

- **IMDb Reviews Dataset** (via `tensorflow_datasets`)
- Binary sentiment classification: Positive or Negative

## 🏗️ Model Architecture

- Embedding + Positional Encoding  
- Multiple Encoder Layers with:
  - Gaussian-Scaled Multi-Head Attention
  - Layer Normalization and Dropout
  - Feedforward Neural Networks  
- GlobalAveragePooling + Dense Sigmoid Output

## 🔬 Custom Attention: ATRT

Introduces a **Gaussian-based scaling** to standard attention weights, encouraging smoother and more stable convergence while preserving performance.




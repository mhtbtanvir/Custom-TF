# ATRT – Adaptive Token Ranking and Transformation

A custom attention-regularized Transformer model built from scratch in TensorFlow for text classification. ATRT introduces a novel adaptive attention mechanism to rank and scale token importance dynamically, improving convergence and generalization.

## 🔍 What is ATRT?

**ATRT** is an adaptive attention mechanism that ranks and transforms token importance using **Gaussian smoothing** based on the global mean (μ) and standard deviation (σ) of attention weights.

### Token Importance Zones

Tokens are categorized into zones with corresponding scaling strategies:

| Zone                | Condition                  | Scaling Strategy        |
|---------------------|----------------------------|-------------------------|
| High Importance     | > μ + 1.5σ                 | Top-down decay (1.25–1.30) |
| Moderate Importance | μ to μ + 1.5σ              | Centered smoothing (1.10–1.15) |
| Baseline            | ≤ μ                        | Flat scaling (1.00)     |

This approach encourages the model to emphasize contextually significant tokens while maintaining global coherence.

## 🛠 Task-Specific Customization

ATRT is flexible and can be tailored to different tasks:

- **Classification:** Stronger emphasis on top tokens (higher scaling in High Importance Zone).
- **Summarization:** Smoother, broader scaling to highlight rare but critical tokens.

Zone boundaries and scaling factors are adjustable based on task needs.

---

## 🧪 Performance Comparison

| Metric                     | Base Transformer       | ATRT Transformer        |
|----------------------------|------------------------|-------------------------|
| **Training Accuracy (Epoch 1)** | 71.90%                 | 86.06%                  |
| **Training Accuracy (Epoch 10)**| 98.79%                | 96.04%                  |
| **Validation Accuracy (Avg)**   | ~95.20% (fluctuates)  | ~94–95% (stable)        |
| **Test Accuracy**              | **94.96%**             | **95.32%**              |

🔎 **Insight:**  
- ATRT achieves **faster convergence in early epochs** and demonstrates **greater stability in validation accuracy**.  
- Despite a slightly lower final training accuracy, ATRT outperforms the base Transformer in **generalization**, as shown by the higher test accuracy.

---

## ⚙️ Key Features

- ✅ Custom Transformer architecture in **TensorFlow**
- 🧠 **Gaussian-scaled multi-head attention** for adaptive token ranking
- 📈 **Improved convergence** and **generalization**
- 🧪 Uses **IMDb** dataset for binary sentiment classification
- 📦 Built with `TextVectorization`, `tf.data`, and modular components in a **Jupyter Notebook**

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/ATRT-Transformer.git
cd ATRT-Transformer
pip install -r requirements.txt  # optional
jupyter notebook imdb_atrt.ipynb

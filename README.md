# Semantic Similarity with Attention

A Natural Language Processing (NLP) project focused on measuring semantic similarity between text pairs using an attention-based neural architecture.

The project explores how attention mechanisms can capture relationships between words and how multiple similarity measures can be fused to improve semantic similarity prediction.

---

## 1. Project Overview

Semantic similarity measures how closely two pieces of text are related in meaning.

For example:

```text
Sentence 1:
"How do I reset my password?"

Sentence 2:
"I forgot my password. How can I change it?"
```

Although the wording is different, both sentences express a similar intent.

This project develops an attention-based model to learn semantic relationships between text pairs and combines multiple similarity measures to produce the final prediction.

---

## 2. Proposed Architecture

The proposed model consists of several components that transform input text into contextual representations, calculate multiple similarity measures, and combine them for the final prediction.

### Architecture

<p align="center">
  <img src="results/architecture/proposed_model.png" width="850"/>
</p>

*Figure 1. Proposed model architecture.*

### Model Pipeline

```text
Input Text Pair
      ↓
Tokenization
      ↓
Embedding
      ↓
Attention Mechanism
      ↓
Contextual Representation
      ↓
Sentence Representation
      ↓
┌───────────────────────────────┐
│       Fusion Similarity       │
│                               │
│  Cosine    Manhattan    Euclidean
└───────────────────────────────┘
      ↓
Fusion Layer
      ↓
Prediction
```

---

## 3. Attention Mechanism

The proposed architecture uses the scaled dot-product attention mechanism.

Given Query, Key, and Value matrices:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

The attention mechanism is defined as:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

where:

* $Q$ = Query matrix
* $K$ = Key matrix
* $V$ = Value matrix
* $d_k$ = dimensionality of the key vectors
* $W_Q$, $W_K$, and $W_V$ = trainable projection matrices

### Multi-Head Attention

For multi-head attention:

$$
\text{MultiHead}(Q,K,V)
=
\text{Concat}(head_1,\ldots,head_h)W_O
$$

where:

$$
head_i =
\text{Attention}
(QW_i^Q,KW_i^K,VW_i^V)
$$

Multiple attention heads allow the model to capture different relationships between tokens.

---

## 4. Attention Visualization

To analyze how the model processes the input text, attention weights are visualized using a heatmap.

The heatmap provides a token-level view of the relationships learned by the attention mechanism.

### Attention Heatmap

<p align="center">
  <img src="results/attention/attention_heatmap.png" width="850"/>
</p>

*Figure 2. Attention heatmap generated from the proposed model.*

Higher attention weights indicate stronger interactions between tokens.

---

## 5. Fusion Similarity

The proposed model does not rely on a single similarity measure.

Instead, it combines three different similarity/distance measures:

* **Cosine Similarity**
* **Manhattan Distance**
* **Euclidean Distance**

Each measure captures a different aspect of the relationship between two sentence representations.

### 5.1 Cosine Similarity

Given two sentence representations $u$ and $v$:

$$
\text{Cosine}(u,v)
=
\frac{u \cdot v}
{\|u\|\|v\|}
$$

Cosine similarity measures the angular similarity between two embeddings.

A higher value indicates that the two representations point in a more similar direction.

---

### 5.2 Manhattan Distance

The Manhattan distance is defined as:

$$
D_{\text{Manhattan}}(u,v)
=
\sum_{i=1}^{n}|u_i-v_i|
$$

It measures the absolute difference between corresponding dimensions of two sentence representations.

A smaller distance indicates that the representations are closer.

---

### 5.3 Euclidean Distance

The Euclidean distance is defined as:

$$
D_{\text{Euclidean}}(u,v)
=
\sqrt{\sum_{i=1}^{n}(u_i-v_i)^2}
$$

It measures the geometric distance between two sentence representations.

A smaller distance indicates greater similarity between the representations.

---

### 5.4 Similarity Fusion

The outputs from the three similarity measures are combined through the fusion layer:

$$
S_{\text{fusion}}
=
f
\left(
S_{\text{cosine}},
D_{\text{Manhattan}},
D_{\text{Euclidean}}
\right)
$$

where $f(\cdot)$ represents the fusion mechanism used by the proposed architecture.

The resulting fused representation is then passed to the prediction layer.

### Fusion Pipeline

```text
Sentence A ──→ Encoder ──→ Embedding A ──┐
                                         │
                                         ├──→ Cosine Similarity ──┐
Sentence B ──→ Encoder ──→ Embedding B ──┤                        │
                                         ├──→ Manhattan Distance ─┤
                                         │                        ├──→ Fusion ──→ Prediction
                                         └──→ Euclidean Distance ─┘
```

The combination of these measures allows the model to capture both **directional similarity** and **distance-based relationships** between sentence embeddings.

---

## 6. Training Performance

Training and validation curves are analyzed to monitor model learning and identify potential overfitting or underfitting.

### Training & Validation Curves

<p align="center">
  <img src="results/training/training_curve.png" width="850"/>
</p>

*Figure 3. Training and validation performance across epochs.*

The curves are used to analyze:

* Training convergence
* Validation performance
* Overfitting
* Underfitting
* Training stability

---

## 7. Model Evaluation

The model is evaluated using classification metrics to measure its ability to distinguish between semantically similar and dissimilar text pairs.

### ROC-AUC

<p align="center">
  <img src="results/evaluation/roc_auc.png" width="700"/>
</p>

*Figure 4. ROC curve and AUC performance.*

ROC-AUC measures the model's ability to distinguish between positive and negative semantic similarity pairs.

---

## 8. Baseline Comparison

The proposed architecture is compared against a baseline model to evaluate the contribution of the attention and fusion mechanisms.

| Model              | Trainable Parameters |   Accuracy |  Precision |     Recall |         F1 |   ROC-AUC |
| ------------------ | -------------------: | ---------: | ---------: | ---------: | ---------: | --------: |
| Baseline           |             `XX,XXX` |   `XX.XX%` |   `XX.XX%` |   `XX.XX%` |   `XX.XX%` |   `XX.XX` |
| **Proposed Model** |           **XX,XXX** | **XX.XX%** | **XX.XX%** | **XX.XX%** | **XX.XX%** | **XX.XX** |

### Model Performance Comparison

<p align="center">
  <img src="results/evaluation/model_comparison.png" width="850"/>
</p>

*Figure 5. Performance comparison between the baseline and proposed model.*

---

## 9. Trainable Parameters

The number of trainable parameters is analyzed to compare the complexity of the proposed model against the baseline.

Trainable parameters are the weights that are updated during the training process.

```python
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

### Parameter Comparison

```text
Baseline Model:
XX,XXX trainable parameters

Proposed Model:
XX,XXX trainable parameters
```

This analysis provides insight into the trade-off between model complexity and predictive performance.

---

## 10. Results Summary

The proposed model is evaluated from three main perspectives:

### Predictive Performance

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Model Complexity

The number of trainable parameters is compared with the baseline to quantify the complexity of the proposed architecture.

### Interpretability

Attention weights are visualized to provide insight into token-level relationships learned by the model.

---

## 11. Key Findings

The experiments investigate whether the proposed architecture can:

* Capture semantic relationships between text pairs
* Utilize attention to learn contextual token relationships
* Combine multiple similarity measures effectively
* Improve predictive performance compared with the baseline
* Maintain a reasonable number of trainable parameters
* Provide interpretable attention patterns

---

## 12. Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 13. Project Structure

```text
Semantic-Similarity-NLP/
│
├── data/
│
├── notebooks/
│   ├── training.ipynb
│   └── attention_visualization.ipynb
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── train.py
│
├── models/
│
├── results/
│   ├── architecture/
│   │   └── proposed_model.png
│   │
│   ├── attention/
│   │   └── attention_heatmap.png
│   │
│   ├── training/
│   │   └── training_curve.png
│   │
│   └── evaluation/
│       ├── roc_auc.png
│       └── model_comparison.png
│
├── requirements.txt
└── README.md
```

---

## 14. Installation

Clone the repository:

```bash
git clone https://github.com/HaikalSyafie/Semantic-Similarity-NLP.git
cd Semantic-Similarity-NLP
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 15. Usage

Run the training pipeline:

```bash
python src/train.py
```

To visualize attention weights, run:

```text
notebooks/attention_visualization.ipynb
```

---

## 16. Future Improvements

* Fine-tune the architecture on larger semantic similarity datasets
* Experiment with different attention configurations
* Analyze attention patterns across different sentence pairs
* Optimize inference performance
* Deploy the model as a REST API using FastAPI

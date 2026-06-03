# ✏️ Generative AI — Synthetic Handwritten Digits (VAE)

A generative deep learning project that trains a **Variational Autoencoder (VAE)** on the MNIST dataset to learn a structured latent space of handwritten digits and synthesize novel, realistic digit images from random noise.

---

## 📌 Project Overview

| | |
|---|---|
| **Goal** | Generate synthetic handwritten digits via a learned latent space |
| **Dataset** | MNIST (60,000 training images, 28×28 grayscale) |
| **Model** | Variational Autoencoder (VAE) |
| **Latent Dimensions** | 20 |
| **Tools** | Python, PyTorch, torchvision, Matplotlib |
| **Hardware** | GPU (T4) |

---

## 🧠 Model Architecture

```
Input Image (28×28 = 784)
        │
   [Encoder FC]   784 → 400 (ReLU)
        │
   ┌────┴────┐
 [fc_mu]  [fc_logvar]    400 → 20  (mean & log-variance)
        │
 [Reparameterize]        z = μ + ε·σ   (ε ~ N(0,1))
        │
   [Decoder FC1]   20 → 400 (ReLU)
        │
   [Decoder FC2]   400 → 784 (Sigmoid)
        │
  Reconstructed Image (28×28)
```

### Key Design Decisions
- **Sigmoid output**: forces all 784 pixel values into [0, 1], making them compatible with Binary Cross-Entropy loss
- **Reparameterization trick**: enables backpropagation through the stochastic sampling step by factoring randomness out as external noise (ε)
- **Log-variance instead of variance**: numerically more stable; always finite and differentiable

---

## 📉 Loss Function

The VAE is trained with a composite loss consisting of two mathematically principled terms:

```
Loss = BCE + KLD
```

**Binary Cross-Entropy (BCE)** — Reconstruction Loss
```
BCE = -Σ [ x·log(x̂) + (1-x)·log(1-x̂) ]
```
Measures pixel-level reconstruction accuracy. Penalizes confident wrong pixel guesses and discourages gray, ambiguous outputs.

**Kullback–Leibler Divergence (KLD)** — Regularization
```
KLD = -0.5 · Σ (1 + log(σ²) - μ² - σ²)
```
Forces every image's latent distribution toward the standard normal N(0,1), ensuring the latent space is smooth, structured, and interpolable — not a disordered set of disconnected clusters.

> ⚠️ Both equations are mathematically proven probability laws and are **not hyperparameters to be tuned**.

---

## 🔧 Training

### Preprocessing
- Images converted to tensors and normalized to [0, 1] via `transforms.ToTensor()`
- Loaded in batches of **32** via `DataLoader` with shuffle enabled

### Two Optimizers Compared

| Optimizer | Epochs | Learning Rate | Notes |
|---|---|---|---|
| Vanilla GD | 10 | 1e-3 | Manual `param.data -= lr * param.grad` |
| **Adam** ✅ | 10 | 1e-3 | Faster convergence, lower loss |

Adam was selected as the final optimizer due to adaptive moment estimation providing more stable and efficient weight updates on this task.

### Training Safeguards
- `torch.autograd.set_detect_anomaly(True)` enabled for informative CUDA tracebacks
- Per-batch NaN/Inf loss detection with batch skipping to handle numerical instabilities

---

## 🎨 Generation

After training, **64 new digits** are synthesized by:

1. Sampling random latent vectors `z ~ N(0, I)` of shape `[64, 20]`
2. Passing them through the trained **decoder only**
3. Reshaping outputs to `[64, 1, 28, 28]` and plotting in an **8×8 grid**

This demonstrates the model has learned a continuous, generative distribution over digit space — not mere memorization.

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/your-username/vae-mnist-digits.git
cd vae-mnist-digits

# Install dependencies
pip install torch torchvision matplotlib

# Launch the notebook
jupyter notebook GenAI_Digits.ipynb
```

> **Note**: A CUDA-capable GPU (e.g. T4) is recommended for faster training. The notebook auto-detects and moves the model to `cuda` if available.

---

## 📁 Repository Structure

```
vae-mnist-digits/
│
├── GenAI_Digits.ipynb   # Full VAE training and generation notebook
├── data/                # MNIST dataset (auto-downloaded on first run)
└── README.md            # Project documentation
```

---

## 🔍 Key Takeaways

- The reparameterization trick is what makes VAEs trainable end-to-end — it moves randomness outside the computational graph
- BCE alone produces sharp but poorly organized latent spaces; KLD regularization is what enables smooth interpolation and coherent generation
- Adam converges significantly faster than vanilla gradient descent on this task due to adaptive per-parameter learning rates
- The 20-dimensional latent space is compact enough to generalize well but expressive enough to capture digit variation across all 10 classes

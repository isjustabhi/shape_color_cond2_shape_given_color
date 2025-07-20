# Conditional Generative Model: P(color) × P(shape | color)

This project implements an **improved conditional generative model** where grayscale shapes are generated based on color input. The model follows the probabilistic factorization:

\[
P(\text{shape}, \text{color}) \approx P(\text{color}) \times P(\text{shape} \mid \text{color})
\]

This work is part of a comparative study inspired by the paper **"Compositional Generative Modeling: A Single Model is Not All You Need"**, focusing on compositionality and modular generative systems.

---

## ✅ **Overview**

- **Goal:** Generate shapes (circle, square, triangle) conditioned on a color input (Red, Green, Blue).
- **Purpose:** Evaluate compositional generalization when conditioning on color.
- **Output:** Grayscale shape for a given color condition.

---

## ✅ **Architecture Improvements**

- **Conditional FiLM Layers:** Strong conditioning of features on color.
- **Beta-VAE Loss:** Encourages disentangled latent space.
- **Edge-Aware Regularization:** Improves shape boundaries and sharpness.
- **Latent Space:** Increased to 64 for better diversity.
- **Decoder:** U-Net-like structure with skip connections for reconstruction quality.

---

## ✅ **Project Structure**

shape_color_cond2_shape_given_color/
├── data/
│   └── toy_dataset/
│       ├── train/                      # Synthetic training images (colored shapes)
│       └── test/                       # Synthetic test images
│
├── models/
│   └── cond_shape_given_color_improved.pth  # Saved model + loss history
│
├── samples/
│   └── cond2_improved/
│       ├── generated_epoch5.png
│       ├── generated_epoch10.png
│       ├── ...
│       └── generated_epoch150.png      # Generated samples (Red | Green | Blue)
│
├── scripts/
│   ├── generate_dataset.py             # Creates synthetic dataset
│   ├── train_shape_given_color_improved.py # Training script with FiLM + Beta-VAE
│   └── view_results.ipynb              # Visualization notebook (loss curve + images)
│
├── shape_dataset.py                    # Dataset loader for grayscale + color condition
├── shape_generator_improved.py         # Improved Conditional VAE (FiLM + U-Net-like)
│
├── requirements.txt                    # Dependencies
├── README.md                           # Full project description, results, usage
└── .gitignore                          # Ignores models, data, cache, etc.

---

---

## ✅ **Installation**

```bash
# Clone repository
git clone https://github.com/isjustabhi/shape_color_cond2_shape_given_color.git
cd shape_color_cond2_shape_given_color

# Install dependencies
pip install -r requirements.txt
```
---
✅ Dataset Preparation
Generate a synthetic dataset of colored shapes:
python scripts/generate_dataset.py
This creates:

bash
Copy
Edit
data/toy_dataset/train/
data/toy_dataset/test/
---

✅ Training the Model
Run the training script:

python scripts/train_shape_given_color_improved.py
Features of the training loop:

Loss Curve: Stored as loss_history in checkpoint.

Samples: Generated every 5 epochs under samples/cond2_improved/.

Checkpoint: Includes model weights + loss history at models/cond_shape_given_color_improved.pth.
---

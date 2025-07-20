Here’s a **clean, concise README.md** version that includes only the essential details for GitHub:

---

# Conditional Generative Model: P(color) × P(shape | color)

This project implements an **improved conditional generative model** for generating shapes based on color input, using **FiLM conditioning**, **Beta-VAE loss**, and **edge-aware regularization**.

---

## ✅ Objective

$$
P(\text{shape}, \text{color}) \approx P(\text{color}) \times P(\text{shape} \mid \text{color})
$$

* **Input:** Color condition (Red, Green, Blue)
* **Output:** Grayscale shape (Circle, Square, Triangle)
* **Goal:** Test compositional generalization for unseen combinations.

---

## ✅ Features

* **FiLM Layers** for strong color-based conditioning
* **Beta-VAE Loss** for disentangled latent space
* **Edge-Aware Regularization** for sharp shapes
* **U-Net-like Decoder** for high-quality reconstructions

---

## ✅ Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/isjustabhi/shape_color_cond2_shape_given_color.git
cd shape_color_cond2_shape_given_color
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python scripts/generate_dataset.py
```

### 3. Train Model

```bash
python scripts/train_shape_given_color_improved.py
```

* Samples saved in: `samples/cond2_improved/`
* Checkpoint: `models/cond_shape_given_color_improved.pth`

### 4. Visualize

```bash
jupyter notebook scripts/view_results.ipynb
```

---

## ✅ Results (Example)

| Epoch | Generated Samples                                          |
| ----- | ---------------------------------------------------------- |
| 10    | ![Epoch10](samples/cond2_improved/generated_epoch10.png)   |
| 150   | ![Epoch150](samples/cond2_improved/generated_epoch150.png) |

---

## ✅ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pillow
matplotlib
numpy
jupyter
```

---

## ✅ License

MIT License © 2025 Abhiram Varma Nandimandalam

```

---

✅ This version is **short, professional, and GitHub-friendly**. It includes:
- Goal
- Key features
- Quick commands for setup
- Results preview
- Requirements

---

Do you want me to now **generate the `view_results.ipynb` notebook** that will:
✔ Plot **loss curve**  
✔ Show **latest generated image**  
✔ Show **real dataset examples**  
✔ Combine all generated epochs into a progression grid?
```

import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shape_dataset import ShapeGivenColorDataset
from shape_generator_improved import ConditionalShapeVAEImproved

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
BATCH_SIZE = 32
LATENT_DIM = 64
SAMPLE_EVERY = 5
BETA = 4.0        # Beta-VAE KL scaling
EDGE_WEIGHT = 0.1 # Edge regularization weight

# Paths
train_path = "data/toy_dataset/train"
os.makedirs("models", exist_ok=True)
os.makedirs("samples/cond2_improved", exist_ok=True)

# Dataset
dataset = ShapeGivenColorDataset(train_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model & Optimizer
model = ConditionalShapeVAEImproved(latent_dim=LATENT_DIM, condition_dim=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss Functions
def vae_loss(x_recon, x, mu, logvar, beta=BETA):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div

def edge_loss(pred, target):
    # Crop to smallest common size
    H = min(pred.shape[2], target.shape[2])
    W = min(pred.shape[3], target.shape[3])
    pred = pred[:, :, :H, :W]
    target = target[:, :, :H, :W]

    # Compute gradient differences
    pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

    # Match sizes
    grad_H = min(pred_grad_y.shape[2], target_grad_y.shape[2])
    grad_W = min(pred_grad_x.shape[3], target_grad_x.shape[3])
    pred_grad_x, target_grad_x = pred_grad_x[:, :, :grad_H, :grad_W], target_grad_x[:, :, :grad_H, :grad_W]
    pred_grad_y, target_grad_y = pred_grad_y[:, :, :grad_H, :grad_W], target_grad_y[:, :, :grad_H, :grad_W]

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

# Training Loop
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for color_vec, gray_img in dataloader:
        color_vec, gray_img = color_vec.to(DEVICE), gray_img.to(DEVICE)

        x_recon, mu, logvar = model(gray_img, color_vec)
        base_loss = vae_loss(x_recon, gray_img, mu, logvar, beta=BETA)
        edge_reg = edge_loss(x_recon, gray_img)
        loss = base_loss + EDGE_WEIGHT * edge_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f}")

    # Save sample generations
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            samples = []
            for i in range(3):  # Red, Green, Blue
                condition = torch.zeros(1, 3).to(DEVICE)
                condition[0, i] = 1
                z = torch.randn(1, LATENT_DIM).to(DEVICE)
                generated = model.decode(z, condition)
                samples.append(generated)

            save_image(torch.cat(samples, dim=0),
                       f"samples/cond2_improved/generated_epoch{epoch+1}.png", nrow=3)

# Save model & loss history
torch.save({
    "model_state": model.state_dict(),
    "loss_history": loss_history
}, "models/cond_shape_given_color_improved.pth")

print(" Improved model and loss history saved!")

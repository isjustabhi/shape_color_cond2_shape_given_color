import torch
import torch.nn as nn
import torch.nn.functional as F

# FiLM Layer for conditioning
class FiLM(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)

    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ConditionalShapeVAEImproved(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Condition embedding
        self.condition_fc = nn.Linear(condition_dim, 64)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + 64, 128 * 8 * 8)
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec_deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

        # FiLM Layers
        self.film1 = FiLM(64, 64)
        self.film2 = FiLM(32, 64)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        cond_emb = F.relu(self.condition_fc(condition))
        z = torch.cat([z, cond_emb], dim=1)
        x = self.fc_decode(z).view(-1, 128, 8, 8)

        x = F.relu(self.dec_deconv1(x))
        x = self.film1(x, cond_emb)

        x = F.relu(self.dec_deconv2(x))
        x = self.film2(x, cond_emb)

        x = torch.sigmoid(self.dec_deconv3(x))
        return x

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        return x_recon, mu, logvar

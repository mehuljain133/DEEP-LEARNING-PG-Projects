# Unit-V Autoencoders: Undercomplete autoencoders, regularized autoencoders, sparse autoencoders, denoising autoencoders, representational power, layer, size, and depth of autoencoders, stochastic encoders and decoders.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Base Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=32, sparse=False, stochastic=False, depth=1):
        super().__init__()
        self.sparse = sparse
        self.stochastic = stochastic

        layers = [nn.Flatten(), nn.Linear(28*28, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            self.decoder_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.decoder_layers += [nn.Linear(hidden_dim, 28*28), nn.Sigmoid()]
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        if self.sparse:
            self.sparse_penalty = torch.mean(torch.abs(x))

        if self.stochastic:
            noise = torch.randn_like(x) * 0.1
            x = x + noise

        z = self.fc_latent(x)
        out = self.decoder(z)
        return out.view(-1, 1, 28, 28)

# Train function
def train(model, noise=False, epochs=5, reg_weight=1e-4, sparse_weight=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            noisy_images = images + 0.3 * torch.randn_like(images) if noise else images
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            if model.sparse:
                loss += sparse_weight * model.sparse_penalty

            l2_reg = sum(torch.norm(p) for p in model.parameters())
            loss += reg_weight * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Plot results
def show_reconstructions(model, num_images=8):
    model.eval()
    imgs, _ = next(iter(train_loader))
    imgs = imgs[:num_images].to(device)
    with torch.no_grad():
        recon = model(imgs)
    imgs = imgs.cpu()
    recon = recon.cpu()
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    for i in range(num_images):
        axs[0, i].imshow(imgs[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    plt.suptitle("Top: Original | Bottom: Reconstructed")
    plt.show()

# 1. Undercomplete AE
print("\n--- Undercomplete Autoencoder ---")
model1 = Autoencoder(latent_dim=16)
train(model1)
show_reconstructions(model1)

# 2. Regularized AE (L2)
print("\n--- Regularized Autoencoder ---")
model2 = Autoencoder(latent_dim=16)
train(model2, reg_weight=1e-2)
show_reconstructions(model2)

# 3. Sparse AE (L1 on hidden activations)
print("\n--- Sparse Autoencoder ---")
model3 = Autoencoder(latent_dim=16, sparse=True)
train(model3)
show_reconstructions(model3)

# 4. Denoising AE
print("\n--- Denoising Autoencoder ---")
model4 = Autoencoder(latent_dim=16)
train(model4, noise=True)
show_reconstructions(model4)

# 5. Stochastic Encoder/Decoder
print("\n--- Stochastic Autoencoder ---")
model5 = Autoencoder(latent_dim=16, stochastic=True)
train(model5)
show_reconstructions(model5)

# 6. Deep AE (depth=3)
print("\n--- Deep Autoencoder (depth=3) ---")
model6 = Autoencoder(latent_dim=16, depth=3)
train(model6)
show_reconstructions(model6)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

class SimpleLogger:
    def __init__(self):
        self.d_losses = []
        self.g_losses = []
        self.epochs = []
        
        # Create plots directory
        os.makedirs('training_plots', exist_ok=True)
    
    def log(self, epoch, d_loss, g_loss):
        self.epochs.append(epoch)
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.d_losses, label='Discriminator Loss', color='blue')
        plt.plot(self.epochs, self.g_losses, label='Generator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('training_plots/training_progress.png')
        plt.close()

class ArtDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = list(Path(folder_path).glob("*.jpg"))
        print(f"Found {len(self.image_files)} images in dataset")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial upsampling
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def train_model(dataloader, num_epochs=100, latent_dim=100, device='cpu'):
    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Initialize logger
    logger = SimpleLogger()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1, 1, 1).to(device)
            label_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
            
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()
            
            # Log progress
            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
                logger.log(epoch + i/len(dataloader), d_loss.item(), g_loss.item())
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake = generator(torch.randn(16, latent_dim, 1, 1, device=device))
                torchvision.utils.save_image(fake.detach(),
                                          f'training_plots/fake_samples_epoch_{epoch+1}.png',
                                          normalize=True)
    
    return generator, discriminator

if __name__ == "__main__":
    # Set up data processing
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Create dataset and dataloader
    dataset = ArtDataset("christmasfreud", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    generator, discriminator = train_model(dataloader, device=device)
    
    print("Training complete! Check training_plots folder for progress plots and generated images.")

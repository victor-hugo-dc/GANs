import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Define the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        x = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images, labels):
        x = torch.cat((self.label_embedding(labels), images.view(images.size(0), -1)), -1)
        x = self.model(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define the loss functions and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

# Define the training loop
def train_step(images, labels):
    batch_size = labels.size(0)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Train discriminator
    discriminator_optimizer.zero_grad()

    real_output = discriminator(images, labels)
    real_loss = criterion(real_output, real_labels)

    noise = torch.randn(batch_size, 100).to(device)
    fake_images = generator(noise, labels)
    fake_output = discriminator(fake_images, labels)
    fake_loss = criterion(fake_output, fake_labels)

    discriminator_loss = real_loss + fake_loss
    discriminator_loss.backward()
    discriminator_optimizer.step()

    # Train generator
    generator_optimizer.zero_grad()

    noise = torch.randn(batch_size, 100).to(device)
    fake_images = generator(noise, labels)
    fake_output = discriminator(fake_images, labels)
    generator_loss = criterion(fake_output, real_labels)

    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item(), discriminator_loss.item()

# Function to save side-by-side comparison images
def save_comparison_images(epoch, writer):
    noise = torch.randn(10, 100).to(device)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
    generated_images = generator(noise, labels)

    real_images = [train_data[i][0] for i in range(10)]
    real_images = torch.stack(real_images).to(device)

    comparison_images = torch.cat((real_images, generated_images), dim=0)
    grid = torchvision.utils.make_grid(comparison_images, nrow=10)
    writer.add_image("comparison", grid, epoch)

# Train the cGAN
def train(epochs):
    writer = SummaryWriter(log_dir='logs/cgan_mnist')
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            generator_loss, discriminator_loss = train_step(images, labels)
            writer.add_scalar("generator_loss", generator_loss, epoch * len(train_loader) + i)
            writer.add_scalar("discriminator_loss", discriminator_loss, epoch * len(train_loader) + i)
        
        save_comparison_images(epoch, writer)

train(100)
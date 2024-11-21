import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_images(original_image, augmented_images, save_path='augmentation_examples.png'):
    """Display original and augmented images"""
    plt.figure(figsize=(15, 3))
    
    # Show original
    plt.subplot(1, 5, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Show augmented versions
    for idx, img in enumerate(augmented_images):
        plt.subplot(1, 5, idx + 2)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Augmented {idx + 1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation and transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Basic transform for visualization
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load dataset
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    
    # Create and save augmentation examples
    example_dataset = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
    example_image = example_dataset[0][0]

    # Generate augmented examples
    augmented_images = []
    for _ in range(4):
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
        ])
        augmented_images.append(aug_transform(example_image))
    
    # Save augmentation examples
    show_augmented_images(example_image, augmented_images)
    
    # Create a subset of 25000 samples
    subset_indices = torch.randperm(len(full_dataset))[:25000]
    train_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print(f"Training with {len(train_dataset)} samples")
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp and device info --
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_type = "CPU" if device.type == "cpu" else "GPU"
    model_path = f'model_{timestamp}_{device_type}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    
if __name__ == "__main__":
    train() 
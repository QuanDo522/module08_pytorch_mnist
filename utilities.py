import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import argparse

class ConvNet(nn.Module):
    """
    This is the class that defines our model architecture and the forward passing of inputs through each layer
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) 
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out

def load_data(batch_size):
    """
    This function takes in a batch_size and using the torchvision library, will go a download the data and preprocess the images into Tensors and perform the batching
    and shuffling of the dataset.
    """
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())

    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    # DataLoader for the test dataset, used for evaluating the model
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    return train_loader, test_loader

def training(model, num_epochs, train_loader, test_loader, criterion, optimizer, device):
    """
    This function controls the main logic behind training up the model. 
    """
    # Train the model
    total_step = len(train_loader)  # Total number of batches
    loss_list = []  # List to store loss values
    acc_train = []  # List to store training accuracy values
    acc_test = []  # List to store testing accuracy values

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        # Loop over each batch in the training DataLoader

        pbar = tqdm(train_loader,
                    desc = f"Training: {epoch + 1:03d}/{num_epochs}",
                    ncols = 125,
                    leave = True)

        # Creating running loss & accuracy empty lists
        running_loss = []
        running_accuracy = []

        for i, (images, labels) in enumerate(pbar, 1):
            # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
            # For example, (100, 1, 28, 28) for the MNIST data
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device

            # Forward pass: compute predicted y by passing x to the model
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

            # Backward pass & optimize
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Add the runnign loss
            pbar.set_postfix(loss = f"{sum(running_loss)/i : 10.6f}")

        # Get the average of all losses in the running_loss as the loss of the current epoch
        loss_list.append(sum(running_loss)/i)

        # Track accuracy on the train set
        acc_train.append(track_accuracy(model, train_loader, device))

        # Track accuracy on the test set
        acc_test.append(track_accuracy(model, test_loader, device))

    return model, loss_list, acc_train, acc_test

def track_accuracy(model, loader, device):
    """
    This function serves as an evaluation stage to get an accuracy metric for the model. 
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Increment total by the number of labels
            correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
        accuracy = 100 * correct / total  # Calculate accuracy
    model.train()  # Set the model back to training mode
    return accuracy

def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    """
    This function is a plotting helper function that will generate a curve detailing model performance over the training steps
    """
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.savefig(save_path)

def myargs():
    """
    This function defines what arguments will be passed in when run.py is called.
    """
    parser = argparse.ArgumentParser(description = "Define hyperparameters for our model")

    parser.add_argument("-b", 
                        "--batch_size",
                        type = int,
                        help = "Batch Size",
                        default = 64)
    
    parser.add_argument("-e",
                        "--num_epochs",
                        type = int,
                        help = "Number of Epochs",
                        default = 25)

    parser.add_argument("-l",
                        "--learning_rate",
                        type = float,
                        help = "Learning Rate",
                        default = 0.001)
    
    return parser.parse_args()

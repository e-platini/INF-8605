import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # First block: 2 convolutional layers + pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second block: 2 convolutional layers + pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third block: 2 convolutional layers + pooling
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 16 * 16, 256)  # Adjust based on input image size and pooling
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.pool4(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 512 * 16 * 16)  # Adjust based on input size
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_model(model, train_loader, val_loader, cuda_device, model_save_path="models/simple_CNN", learning_rate=0.001):
    # USER PARAMETERS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 1000
    max_patience = 3
    max_lr_changes = 3
    threshold_patience = 0.005
    ratio_lr_change = 0.1
    # ratio_threshold_patience_change = 0.5

    train_loss_plot = []
    val_loss_plot = []
    # Just so it is always higher than the first actual loss...
    best_val_loss = 9999
    patience_counter = max_patience

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available, good luck")
    else:
        device = torch.device(cuda_device)

    model.to(device)

    # Training loop
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Training for epoch {epoch}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_plot.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}')

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients needed
            for inputs, labels, _ in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_plot.append(val_loss)

        print(f'Train loss: {epoch_loss}')
        print(f'Val loss: {val_loss}')

        if val_loss < best_val_loss:
            if best_val_loss - val_loss > threshold_patience:
                patience_counter = max_patience
            else:
                patience_counter -= 1
                print(f"Progress on val loss was inferior to {threshold_patience}, patience reduced to {patience_counter}.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter -= 1
            print(f"No progress on val loss was made, patience reduced to {patience_counter}.")

        if patience_counter == 0:
            if max_lr_changes == 0:
                print(f'Best val loss: {best_val_loss}')
                print(f"Stopping training.")
                plt.plot(train_loss_plot, label='Train Loss')
                plt.plot(val_loss_plot, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train and Validation Loss')
                plt.legend()
                plt.show()
                break
            else:
                learning_rate = learning_rate * ratio_lr_change
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                patience_counter = max_patience
                max_lr_changes -= 1
                print(f"Patience dropped to 0, changed learning rate to {learning_rate}.")

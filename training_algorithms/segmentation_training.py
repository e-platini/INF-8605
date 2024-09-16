import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt


def train_segmentation_model(model, train_loader, val_loader, cuda_device, model_save_path="models/simple_CNN", learning_rate=0.001):
    # USER PARAMETERS
    criterion = nn.BCEWithLogitsLoss()
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
    epoch_best_val_loss = -1
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

        for inputs, labels in tqdm(train_loader, desc=f"Training for epoch {epoch}"):
            inputs = inputs.to(device)
            labels = labels.squeeze(3).to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs).squeeze(1)  # Forward pass
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
            for inputs, labels in val_loader:

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
                print(f"Progress on best val loss ({best_val_loss}) was inferior to {threshold_patience}, patience reduced to {patience_counter}.")
            best_val_loss = val_loss
            epoch_best_val_loss = epoch
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter -= 1
            print(f"No progress on best val loss ({best_val_loss}) was made, patience reduced to {patience_counter}.")

        if patience_counter == 0:
            if max_lr_changes == 0:
                print(f'Best val loss: {best_val_loss} at epoch {epoch_best_val_loss}')
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
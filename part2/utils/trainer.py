import numpy as np
import time


def training(model, criterion, optimizer, train_loader, validation_loader, epochs,device):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for it in range(epochs):
        start_time = time.time()
        model.train()
        batch_train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            batch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        batch_validation_loss = []
        for validation_inputs, validation_targets in validation_loader:
            validation_inputs, validation_targets = validation_inputs.to(device), validation_targets.to(device)  # Move test inputs and targets to the GPU
            validation_outputs = model(validation_inputs)
            loss = criterion(validation_outputs, validation_inputs)
            batch_validation_loss.append(loss.item())

        train_loss = np.mean(batch_train_loss)
        validation_loss = np.mean(batch_validation_loss)

        train_losses[it] = train_loss
        validation_losses[it] = validation_loss
        end_time = time.time()
        time_taken = end_time - start_time

        if (it + 1) % 2 == 0:
            print(f'Epoch {it + 1}/{epochs}, Loss_train: {train_loss:.4f}, validation_loss: {validation_loss:.4f}, Time taken: {time_taken:.5f} seconds')

    return train_losses, validation_losses

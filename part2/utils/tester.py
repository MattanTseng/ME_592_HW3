import numpy as np


def testing(model, criterion, test_loader, device):
    model.eval()
    model.to(device)
    batch_test_loss = []
    for test_inputs, test_targets in test_loader:
        # Move test inputs and targets to the GPU
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        loss = criterion(test_outputs, test_inputs)
        batch_test_loss.append(loss.item())
    test_loss = np.mean(batch_test_loss)

    print(f'Loss_test: {test_loss:.4f}')

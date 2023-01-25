import torch
import numpy as np


def validate_single_batch(model, batch, label):
    model.eval()
    with torch.no_grad():
        top_1_accuracy = []
        prediction = model(batch)
        prediction = prediction.argmax(dim=1)
        top_1 = prediction == label
        top_1 = top_1.type(torch.float)
        top_1_accuracy.append(top_1.mean().cpu())
    return np.mean(top_1_accuracy)


def validate_model(model, validation_set_loader, loss_function, device):
    model.eval()
    with torch.no_grad():
        top_1_accuracy = []
        validation_loss = []
        for batch, labels in validation_set_loader:
            labels = labels.to(device)
            predictions = model(batch.to(device))
            loss = loss_function(predictions, labels)
            validation_loss.append(loss.cpu())
            predictions = predictions.argmax(dim=1)
            top_1 = predictions == labels
            top_1 = top_1.type(torch.float)
            top_1_accuracy.append(top_1.mean().cpu())
    return np.mean(top_1_accuracy), np.mean(validation_loss)

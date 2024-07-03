import torch
import math
from sklearn.metrics import confusion_matrix

def compute_accuracy(model: torch.nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def compute_confusion_matrix(model, x_data, y_data, device):
    model.eval()
    all_preds, all_labels = [], []

    batch_size = 32
    n_batches = math.ceil(x_data.shape[0] / batch_size)
     
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x_data[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
            y_curr = y_data[counter * batch_size:(counter + 1) *
                        batch_size].to(device)

            output = model(x_curr)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_curr.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm
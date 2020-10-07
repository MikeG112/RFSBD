# External imports
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             precision_score, recall_score, roc_curve,
                             fbeta_score)

def train(model, train_data, val_data, optimizer, loss_fn,
          epochs, device='cpu', batch_size = 1):

    # Load training data
    train_loader = DataLoader(train_data,
                              batch_size = batch_size,
                              shuffle = True)
    val_loader = DataLoader(val_data,
                            batch_size = batch_size,
                            shuffle = False)

    for epoch in range(1, epochs + 1):

        # Track grads
        model.train()

        # Loss tally
        epoch_train_loss = 0

        for batch_index, batch in enumerate(train_loader):

            # Extract videos, frame_labels
            batch_videos = batch[0].to(device)
            true_labels = batch[1].to(device)

            # Zero accumulated grads
            optimizer.zero_grad()

            # Evaluate model on batch
            outputs = model(batch_videos)

            # Calculate loss
            loss = loss_fn(outputs.squeeze(0), true_labels.squeeze(0).squeeze(1).long())
            epoch_train_loss += loss.item()
            if (batch_index == 0) or (batch_index % 100 == 0):
                print(f"Batch index: {batch_index} loss: {loss:.2f}", sep='')

            # Chain rule
            loss.backward()

            # Descent step
            optimizer.step()

        if (epoch == 1) or (epochs % 10 == 0):
            print(f"Epoch: {epoch}, Loss per batch: {epoch_train_loss/len(train_loader):.2f}",
                  sep='')

        # Validation step
        # TODO: alter evaluation step to fit this pipeloutputs, true_labels.long()ine
        # evaluate_epoch(model, val_loader, val_data,
                        #loss_fn, epoch, device=device)

    return None


def evaluate_epoch(model, val_loader, val_data, loss_fn, epoch, device='cpu'):
    '''
    Evaluates a model on a validation set according to a loss function,
    and additional specified metrics.

    Author's note: This is taken from one of my own private repositories.
    '''

    model.eval()

    epoch_val_loss = 0
    all_outputs = []

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):

            # Extract images, labels
            batch_videos = batch[0].to(device)
            true_labels = batch[1].to(device)

            # Evaluate model on batch
            outputs = model(batch_videos)

            # Calculate loss
            loss = loss_fn(outputs.squeeze(0),
                           true_labels.squeeze(0).squeeze(1).long())
            epoch_val_loss += loss.item()

            # Store outputs for global metric calculation
            all_outputs.append(outputs.squeeze(0))

    # Convert outputs to numpy array
    all_outputs = torch.cat(all_outputs, dim=0)
    all_outputs= all_outputs.reshape(len(all_outputs),-1).cpu()

    all_true_labels = torch.stack([val_data[index][1] for index in range(len(val_data))], dim=0).cpu()

    # Loss
    avg_loss_per_batch = epoch_val_loss / len(val_loader)

    # Multi-class Score based metrics
    try:
        roc_auc = roc_auc_score(all_true_labels, nn.Softmax(dim=1)(all_outputs), multi_class = 'ovr')
    except ValueError:
        roc_auc = 0

    # Convert outputs to labels
    _, all_output_labels = all_outputs.max(dim=1)
    print(len(all_output_labels[all_output_labels==0]))

    # Label based metrics
    accuracy = accuracy_score(all_true_labels, all_output_labels)
    precision = precision_score(all_true_labels, all_output_labels, average = None)
    recall = recall_score(all_true_labels, all_output_labels, average = None)
    f_beta = fbeta_score(all_true_labels, all_output_labels, beta = 1, average = None)

    print(f"Epoch: {epoch}\n",
          f"Average batch loss: {avg_loss_per_batch:.2f}\n",
          f"ROC AUC: {roc_auc:.2f}\n",
          f"Accuracy: {accuracy:.2f}\n",
          f"Precision: {precision}\n",
          f"Recall: {recall}\n",
          f"F_beta: {f_beta}\n",
          sep='')
    return None

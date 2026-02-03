import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#import auc_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

def extract_output(output):
    #if the output has pooler attribute, use it
    if hasattr(output, 'pooler_output'):
        print(output.pooler_output.shape)
        return output.pooler_output
    return output

#use autocast for mixed precision training
def train_loop(model, loader, optimizer, criterion, device, use_amp, gradient_clip=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        img, label = batch
        img = img.to(device)
        label = label.to(device)
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(img)
                output = extract_output(output)
                loss = criterion(output, label)
        else:
            output = model(img)
            output = extract_output(output)
            loss = criterion(output, label)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    logits = [] 
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader):
            img, label = batch
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            output = extract_output(output)
            loss = criterion(output, label)
            total_loss += loss.item()
            for i in range(len(label)):
                logits.append(output[i].detach().cpu())
                targets.append(label[i].detach().cpu())
    return total_loss/len(loader), torch.stack(logits), torch.stack(targets)
def get_metrics(logits, targets):
    preds = torch.argmax(logits, dim=1)
    targets = targets.cpu().numpy()
    #apply softmax to the logits
    logits = torch.nn.functional.softmax(logits, dim=1)
    logits = logits.cpu().numpy()
    preds = preds.cpu().numpy()
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='macro', zero_division=0)
    recall = recall_score(targets, preds, average='macro', zero_division=0)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    # Handle binary vs multiclass AUC calculation
    if logits.shape[1] == 2:  # Binary classification
        logits = logits[:, 1]  # Use probabilities for the positive class
        auc = roc_auc_score(targets, logits)
    else:  # Multiclass classification
        auc = 0
        for i in range(logits.shape[1]):
            y_true_binary = (targets == i).astype(float)  # Convert targets to binary (1 vs rest)
            y_score_binary = logits[:, i]  # Get the predicted scores for the current class
            auc += roc_auc_score(y_true_binary, y_score_binary)
        auc = auc / logits.shape[1]  # Average AUC across all classes

    #round the metrics to 4 decimal places
    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    auc = round(auc, 4)

    return accuracy, precision, recall, f1, auc
  
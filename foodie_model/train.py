import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD

from tqdm import tqdm

from model import ResNet50
from dataloader import dataloader

from utils import plot_loss_accuracy, calculate_accuracy

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = 1e-3
    epochs = 100
    batch_size = 32
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader, val_dataloader, test_dataloader, num_classes = dataloader(batch_size)

    best_model_path = "resnetfoodie.pth"

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    if os.path.exists(best_model_path):
        model = ResNet50(num_classes)
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    else:
        model = ResNet50(num_classes)
        print("Training a new model.")

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_acc = 0.0

    model.to(device)
    for epoch in tqdm(range(epochs), position=0):
        train_loss, train_acc = 0, 0
        model.train()  

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_acc += calculate_accuracy(F.softmax(y_pred, dim=1), y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            val_loss, val_acc = 0, 0
            model.eval()  
            with torch.inference_mode():  
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    val_acc += calculate_accuracy(F.softmax(y_pred, dim=1), y)
                    val_loss += loss.item()

            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            train_acc /= len(train_dataloader)
            val_acc /= len(val_dataloader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(f"Epoch: {epoch} | Train Loss: {train_loss:.2f} Train Accuracy: {train_acc:.2f} | Val Loss: {val_loss:.2f} Val Accuracy: {val_acc:.2f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print("Best model saved!")

    test_loss, test_acc = 0, 0
    model.eval()  
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_acc += calculate_accuracy(F.softmax(y_pred, dim=1), y)
            test_loss += loss.item()
    
    print(f"Test Loss: {test_loss/len(test_dataloader):.2f} Test Accuracy: {test_acc/len(test_dataloader):.2f}")

    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save=True)

if __name__ == "__main__":
    main()

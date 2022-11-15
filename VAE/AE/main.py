
import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from VAE.dataset import MNIST_Dataset
from torch.utils.data import DataLoader
from VAE.utils import pil_to_tensor
from VAE.AE.model import AutoEncoder

def train(model, train_dataloader, val_dataloader, n_epochs, model_out_path):
    """
    Function to train our autoencoder.
    Args:
        model   : object instance of our autoencoder.
        n_epoch : number of epochs to train.
    """
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-05)

    best_val_loss = 99999999
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            img, label = batch
            img = img.squeeze(1).reshape(-1, 28*28)

            optimizer.zero_grad()
            _, reconstructed_img = model(img)

            #Calculate loss
            loss = criterion(reconstructed_img, img)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            epoch_loss += loss
            
        epoch_loss = epoch_loss / i
        val_loss = 0
        model.eval()
        for i, batch in enumerate(val_dataloader):
            img, label = batch
            img = img.squeeze(1).reshape(-1, 28*28)
            
            _, reconstructed_img = model(img)
            #Calculate loss
            loss = criterion(reconstructed_img, img)
            val_loss += loss
            
        val_loss = val_loss/i
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, f'{model_out_path}/best_model.pth')
        if epoch % 10 == 0 or epoch == n_epochs-1:
            print(f"Epoch:{epoch+1}, Train_Loss:{epoch_loss}, Val_Loss:{val_loss}")


def main():
    transforms = [pil_to_tensor]
    train_dataset = MNIST_Dataset(root_dir="~/Anurag/MNIST", 
                                  set='Train', 
                                  transforms=transforms)

    tr_dataloader = DataLoader(train_dataset, batch_size=128,
                               shuffle=True, num_workers=0)

    valid_dataset = MNIST_Dataset(root_dir="~/Anurag/MNIST", 
                                  set='Valid', 
                                  transforms=transforms)

    va_dataloader = DataLoader(valid_dataset, batch_size=128,
                               shuffle=True, num_workers=0)

    
    ae = AutoEncoder(embed_dim=2)
    
    model_out_path = Path("~/Anurag/Projects/VAE/AE/outputs/")
    os.makedirs(model_out_path, exist_ok=True)

    train(ae, tr_dataloader, va_dataloader, 25, model_out_path)

if __name__=='__main__':
    main()


import torch
import torch.nn as nn

class KL_Divergence(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, sigma):
        kl = 0.5*(sigma**2 + mu**2 - torch.log(sigma**2) - 1).sum()
        return kl

class ELBO(nn.Module):
    """
    This class calculates the ELBO loss for training VAE. 
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.kl_div = KL_Divergence()

    def forward(self, input, reconstructed_input, mu, sigma):
        """
        Args:
            input : input img in shape (batch, 28*28)
            reconstructed_input : output of the encoder 
                                  of shape (batch, 28*28)
            mu    : predicted mean of latents.
            sigma : predicted var of latents.
        """
        mse_loss = self.mse(input, reconstructed_input)
        kl_loss = self.kl_div(mu, sigma)
        return mse_loss + kl_loss
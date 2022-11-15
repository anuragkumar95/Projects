import torch.nn as nn

class Encoder(nn.Module):
    """Encoder class for Auto-Encoder"""
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, embed_dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Takes input from MNIST dataset for toy example.
        Args:
            x is the input image in the shape of (batch, 28*28)

        Returns:
            An embedding vector tensor with shape (batch, self.embed_dim).
        """
        x = self.relu(self.linear1(x))
        encoded = self.sig(self.linear2(x))
        return encoded
    
class Decoder(nn.Module):
    """Decoder class for Auto-Encoder"""
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, 512)
        self.linear2 = nn.Linear(512, 28*28)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, embed):
        """
        Takes input the encoded vector.
        Args:
            embed is the input encoded embedding in the shape of (batch, self.embed_dim)

        Returns:
            A flattened tensor of the shape (batch, 28*28).
        """
        x = self.relu(self.linear1(embed))
        decoded = self.relu(self.linear2(x))
        return decoded
    
    
class AutoEncoder(nn.Module):
    """Aute Encoder class"""
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed
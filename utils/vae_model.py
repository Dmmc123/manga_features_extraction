import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class MangaModel(nn.Module):
    def __init__(self, n_size):
        super().__init__()
        self.n_size = n_size
        # Encoder model
        self.encoder = torch.nn.Sequential( # shape (1, 64, 64)
            torch.nn.Conv2d(1, 32, 4, stride=2, padding=1), # (32, 32, 32)
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, 4, stride=2, padding=1), # (32, 16, 16)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1), # (64, 8, 8)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, 4, stride=2, padding=1), # (64, 4, 4)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 256, 4, stride=1), # (256, 1, 1)
            torch.nn.ReLU(),
            torch.nn.Dropout2d(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Flatten(),
            torch.nn.Linear(256, n_size)
        )
        # Decoder model
        self.decoder = torch.nn.Sequential( # shape (64, 28, 28)
            torch.nn.Linear(n_size, 256),
            View((-1, 256, 1, 1)),
            torch.nn.ConvTranspose2d(256, 64, 4, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        # mu, sigma = x[:, :self.n_size], x[:, self.n_size:]
        # x = self.sample(mu, sigma)
        out = self.decoder(x)
        return out
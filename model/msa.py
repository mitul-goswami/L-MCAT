import torch.nn as nn

class ModalitySpectralAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4, embed_dim, kernel_size=1)
        )
        
    def forward(self, x):
        return self.adapter(x)

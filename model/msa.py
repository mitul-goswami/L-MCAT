import torch
import torch.nn as nn

class ModalitySpectralAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        
        super().__init__()
        self.compression = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(4, embed_dim, kernel_size=1)
        )
        self.param_count = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        return self.compression(x)
    
    def __repr__(self):
        return f"MSA(in={self.compression[0].in_channels}, params={self.param_count/1e3:.1f}K)"

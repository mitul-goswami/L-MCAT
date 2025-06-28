import torch
import torch.nn as nn
from .msa import ModalitySpectralAdapter
from .umaa import UMAALayer

class LMCAT(nn.Module):
    def __init__(self, sar_channels=2, optical_channels=10, 
                 num_classes=11, embed_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        
        self.sar_msa = ModalitySpectralAdapter(sar_channels, embed_dim)
        self.optical_msa = ModalitySpectralAdapter(optical_channels, embed_dim)
        
        self.umaa_layers = nn.ModuleList([
            UMAALayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, sar, optical):
        sar_tokens = self.sar_msa(sar).flatten(2).permute(0, 2, 1) 
        optical_tokens = self.optical_msa(optical).flatten(2).permute(0, 2, 1)
        
        total_loss = 0.0
        modalities = [sar_tokens, optical_tokens]
        for layer in self.umaa_layers:
            modalities, layer_loss = layer(modalities)
            total_loss += layer_loss
        
        combined = torch.cat(modalities, dim=1)  
        
        logits = self.classifier(combined)
        return logits, total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class UMAALayer(nn.Module):
    def __init__(self, dim, num_heads=4, temperature=0.07):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temp = temperature
        self.head_dim = dim // num_heads
        
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, modalities):
       
        B = modalities[0].size(0)
        updated_modalities = []
        total_loss = 0.0
        
        for modality in modalities:
            
            qkv = self.qkv_proj(modality).reshape(B, -1, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)  
            
            modality_updates = torch.zeros_like(modality)
            for i, (qi, ki, vi) in enumerate(zip(q, k, v)):
                for j, (qj, kj, vj) in enumerate(zip(q, k, v)):
                   
                    attn_logits = (qi @ kj.transpose(-2,-1)) / (self.head_dim ** 0.5)
                    attn = F.softmax(attn_logits, dim=-1)
                    
                    
                    if i != j:
                        diag_logits = attn_logits.diagonal(dim1=-2, dim2=-1)
                        logits = attn_logits / self.temp
                        contrast_loss = -diag_logits + torch.logsumexp(logits, dim=-1)
                        total_loss += contrast_loss.mean()
                    
                   
                    modality_updates[i] += (attn @ vj).transpose(1,2).reshape(B, -1, self.dim)
            
           
            modality = modality + modality_updates
            modality = self.norm1(modality)
            modality = modality + self.mlp(modality)
            modality = self.norm2(modality)
            updated_modalities.append(modality)
        
        return updated_modalities, total_loss / (len(modalities) * (len(modalities)-1))

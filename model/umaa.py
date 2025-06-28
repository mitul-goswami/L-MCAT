import torch
import torch.nn as nn
import torch.nn.functional as F

class UMAALayer(nn.Module):

    def __init__(self, dim, num_heads=4, temperature=0.07):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, modalities):
       
        total_loss = 0.0
        updated_modalities = []
        
        for modality in modalities:
            B, N, _ = modality.shape
            
         
            q = self.q_proj(modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            
            attn_updates = torch.zeros_like(modality)
            modality_loss = 0.0
            
            for i in range(len(modalities)):
                for j in range(len(modalities)):
                   
                    sim = torch.matmul(q[i], k[j].transpose(-2, -1)) / self.head_dim**0.5
               
                    attn = F.softmax(sim, dim=-1)
                
                    if i != j:
                        diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
                        logits = sim / self.temperature
                        loss = -diag_sim + torch.logsumexp(logits, dim=-1)
                        modality_loss += loss.mean()
                    
                    attn_update = torch.matmul(attn, v[j])
                    attn_update = attn_update.transpose(1, 2).contiguous().view(B, N, self.dim)
                    attn_updates += attn_update
            
            modality = modality + attn_updates
            modality = self.norm1(modality)
            modality = modality + self.mlp(modality)
            modality = self.norm2(modality)
            
            updated_modalities.append(modality)
            total_loss += modality_loss / (len(modalities) * (len(modalities)-1))
        
        return updated_modalities, total_loss

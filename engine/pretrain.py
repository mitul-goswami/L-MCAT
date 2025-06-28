import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.logger import MetricLogger
from model import LMCAT

def pretrain_lmcat(cfg, device):
    model = LMCAT(
        sar_channels=cfg.DATA.SAR_CHANNELS,
        optical_channels=cfg.DATA.OPTICAL_CHANNELS,
        embed_dim=cfg.MODEL.EMBED_DIM,
        num_layers=cfg.MODEL.NUM_LAYERS
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=cfg.OPTIM.LR, 
                           betas=cfg.OPTIM.BETAS,
                           weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    
    train_loader = get_dataloader(cfg, mode='pretrain')
    logger = MetricLogger(cfg.OUTPUT_DIR)
    
    model.train()
    for epoch in range(cfg.OPTIM.EPOCHS):
        for batch_idx, (sar, optical) in enumerate(train_loader):
            sar, optical = sar.to(device), optical.to(device)
            
            
            _, loss = model(sar, optical)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
            if batch_idx % 100 == 0:
                logger.log_step(epoch, epoch * len(train_loader) + batch_idx)
                
        if epoch % 10 == 0:
            ckpt_path = f"{cfg.OUTPUT_DIR}/ckpt_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
    
    torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/pretrained.pth")
    return model

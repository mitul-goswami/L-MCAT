import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

from data.sen12ms import SEN12MS
from data.transforms import TransformCompose, RandomRotate, ColorJitter
from model.lmcat import LMCAT
from utils.logger import MetricLogger
from utils.misc import set_seed, save_checkpoint

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg['SEED'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(cfg['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transforms = TransformCompose([
        RandomRotate(),
        ColorJitter(brightness=0.1)
    ])
    dataset = SEN12MS(
        root=cfg['DATA']['ROOT'],
        split='train',
        patch_size=cfg['DATA']['PATCH_SIZE'],
        transform=transforms
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg['OPTIM']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4
    )
    
    model = LMCAT(
        sar_channels=2,
        optical_channels=10,
        embed_dim=128,
        num_classes=cfg['DATA']['NUM_CLASSES'],
        num_layers=4,
        num_heads=4
    ).to(device)
    
    if 'PRETRAINED' in cfg['MODEL']:
        model.load_state_dict(torch.load(cfg['MODEL']['PRETRAINED']), strict=False)
    
    if cfg['MODEL'].get('FROZEN', False):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optim = Adam(
        model.parameters(),
        lr=cfg['OPTIM']['LR'],
        betas=cfg['OPTIM']['BETAS']
    )
    
    logger = MetricLogger(output_dir, 'finetune')
    
    for epoch in range(cfg['OPTIM']['EPOCHS']):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['OPTIM']['EPOCHS']}")
        
        for sar, optical, labels in pbar:
            sar, optical, labels = sar.to(device), optical.to(device), labels.to(device)
            
            logits, _ = model(sar, optical)
            loss = criterion(logits, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            logger.log('loss', loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'config': cfg
        }, output_dir / f'checkpoint_{epoch}.pth')
    
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    logger.finalize()

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from data.sen12ms import SEN12MS
from data.transforms import TransformCompose, RandomRotate, GaussianNoise
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
        GaussianNoise(sigma=0.02)
    ])
    dataset = SEN12MS(
        root=cfg['DATA']['ROOT'],
        split='train',
        bands=cfg['DATA']['BANDS'],
        patch_size=cfg['DATA']['PATCH_SIZE'],
        num_samples=cfg['DATA']['NUM_SAMPLES'],
        transform=transforms
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg['OPTIM']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4
    )
    
    model = LMCAT(
        sar_channels=len(cfg['DATA']['BANDS']['SAR']),
        optical_channels=len(cfg['DATA']['BANDS']['OPTICAL']),
        embed_dim=cfg['MODEL']['EMBED_DIM'],
        num_layers=cfg['MODEL']['NUM_LAYERS'],
        num_heads=cfg['MODEL']['NUM_HEADS']
    ).to(device)
    
    optim = AdamW(
        model.parameters(),
        lr=cfg['OPTIM']['LR'],
        betas=cfg['OPTIM']['BETAS'],
        weight_decay=cfg['OPTIM']['WEIGHT_DECAY']
    )
    
    logger = MetricLogger(output_dir, 'pretrain')
    
    for epoch in range(cfg['OPTIM']['EPOCHS']):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['OPTIM']['EPOCHS']}")
        
        for sar, optical in pbar:
            sar, optical = sar.to(device), optical.to(device)
            
            _, loss = model(sar, optical)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            logger.log('loss', loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        if (epoch+1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'config': cfg
            }, output_dir / f'checkpoint_{epoch}.pth')
    
    torch.save(model.state_dict(), output_dir / 'pretrained.pth')
    logger.finalize()

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

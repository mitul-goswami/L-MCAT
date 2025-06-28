import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm
from pathlib import Path

from data.sen12ms import SEN12MS
from model.lmcat import LMCAT
from utils.misc import set_seed

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SEN12MS(
        root=cfg['DATA']['ROOT'],
        split=cfg['DATA']['SPLIT'],
        patch_size=cfg['DATA']['PATCH_SIZE']
    )
    loader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    model = LMCAT(
        sar_channels=2,
        optical_channels=10,
        num_classes=11,
        embed_dim=128,
        num_layers=4,
        num_heads=4
    ).to(device)
    model.load_state_dict(torch.load(cfg['MODEL']['CHECKPOINT']))
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for sar, optical, labels in tqdm(loader):
            sar, optical = sar.to(device), optical.to(device)
            logits, _ = model(sar, optical)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    results = {}
    if 'accuracy' in cfg['METRICS']:
        results['accuracy'] = accuracy_score(all_labels, all_preds)
    if 'f1' in cfg['METRICS']:
        results['f1'] = f1_score(all_labels, all_preds, average='weighted')
    if 'kappa' in cfg['METRICS']:
        results['kappa'] = cohen_kappa_score(all_labels, all_preds)
    
    output_dir = Path(cfg['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.txt', 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class MetricLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, experiment_name))
        self.metrics = {}
        self.start_time = time.time()
        
    def log(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.writer.add_scalar(name, value, len(self.metrics[name]))
        
    def log_step(self, epoch, step):
        log_str = f"Step {step} | "
        for name, values in self.metrics.items():
            log_str += f"{name}: {values[-1]:.4f} | "
        print(log_str)
        
    def finalize(self):
        with open(os.path.join(self.log_dir, 'final_metrics.txt'), 'w') as f:
            for name, values in self.metrics.items():
                f.write(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}\n")
        
        total_time = time.time() - self.start_time
        with open(os.path.join(self.log_dir, 'training_time.txt'), 'w') as f:
            f.write(f"Total training time: {total_time/3600:.2f} hours\n")
        
        self.writer.close()

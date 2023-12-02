import numpy as np
import torch
import pytorch_lightning as pl


SEED = 42
# Setting the seed
pl.seed_everything(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("medium")
print("Device:", device)

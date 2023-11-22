"""
    This module stores initiates all the required constants and seeds everything
"""
import numpy as np
import torch
import pytorch_lightning as pl

#####################################################################
#                        HYPERPARAMETERS                            #
#####################################################################

DATA_DIR = "data\\HBW\\"
IMAGES_DIR = "data\\HBW\\images\\val\\"
YAML_PATH = "data\\HBW\\genders.yaml"
CSV_PATH = "data\\HBW\\dataset.csv"
CHECKPOINT_PATH = "model_ckpts\\"
SMPLX_VERTICES_PATH = "data\\HBW\\smplx\\val\\"
SMPLX_PATH = "data\\models\\"

# DATA_DIR = "..\\data\\HBW\\"
# IMAGES_DIR = "..\\data\\HBW\\images\\val\\"
# YAML_PATH = "..\\data\\HBW\\genders.yaml"
# CSV_PATH = "..\\data\\HBW\\dataset.csv"
# CHECKPOINT_PATH = "..\\saved_models\\"
# SMPLX_VERTICES_PATH = "..\\data\\HBW\\smplx\\val\\"
# SMPLX_PATH = "..\\data\\models\\"

IMG_SIZE = 768
STRIP_THICKNESS = 32
NUM_HEADS = 8
NUM_BETAS = 10
NUM_LAYERS = 4
EMBED_SIZE = 512
HIDDEN_SIZE = 2048
BATCH_SIZE = 8
IMG_SEGMENT_TYPE = "strips"

EPOCHS = 300
LR = 3e-3
DROPOUT = 0.0
LR_MILESTONES = [100, 150, 200]

#####################################################################


SEED = 42
# Setting the seed
pl.seed_everything(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

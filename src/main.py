from utils.consts import (
    YAML_PATH,
    IMAGES_DIR,
    SMPLX_PATH,
    SMPLX_VERTICES_PATH,
    CSV_PATH,
)
from utils.consts import (
    BATCH_SIZE,
    IMG_SIZE,
    IMG_SEGMENT_TYPE,
    NUM_BETAS,
    NUM_HEADS,
    NUM_LAYERS,
    STRIP_THICKNESS,
    EMBED_SIZE,
    HIDDEN_SIZE,
    LR,
    DROPOUT,
    LR_MILESTONES,
)
from utils.dataset_utils import create_hbw_csv, get_datasets, get_dataloader
from train import train_model

from argparse import ArgumentParser
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger

import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.getcwd())


def setup_WandB_Logger():
    wandb_logger = WandbLogger(
        name="sanity_run",
        project="HBW Human Shape Estimation",
        log_model=True,
        checkpoint_name="model_ckpts\\",
        save_dir="wandb-logs\\",
    )
    return wandb_logger


def main():
    _ = create_hbw_csv(YAML_PATH, IMAGES_DIR, SMPLX_VERTICES_PATH, CSV_PATH)
    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )
    trainset, testset = get_datasets(CSV_PATH, transform, test_size=0.1)
    trainloader, valloader = get_dataloader(
        trainset, BATCH_SIZE, type="train", val_size=0.2
    )
    testloader = get_dataloader(testset, BATCH_SIZE, type="test")
    model_kwargs = {
        "segment_size": STRIP_THICKNESS,
        "num_layers": NUM_LAYERS,
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "img_size": IMG_SIZE,
        "num_channels": 3,
        "output_size": NUM_BETAS,
        "segment_type": IMG_SEGMENT_TYPE,
        "dropout": DROPOUT,
    }
    wandb_logger = setup_WandB_Logger()
    training_args = [model_kwargs, SMPLX_PATH, BATCH_SIZE, NUM_BETAS, LR, LR_MILESTONES]
    model, result = train_model(
        training_args, trainloader, valloader, testloader, wandb_logger
    )
    print(result)


if __name__ == "__main__":
    main()

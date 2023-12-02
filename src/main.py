from utils import get_data_configs, get_path_configs, get_train_configs
from utils.dataset_utils import (
    create_hbw_csv,
    get_datasets,
    get_dataloader,
    build_transforms,
)
from train import train_model
from pytorch_lightning.loggers import WandbLogger
import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.getcwd())


def main():
    path_cfg = get_path_configs()
    train_cfg = get_train_configs()
    data_cfg = get_data_configs()
    _ = create_hbw_csv(
        path_cfg.yaml_file, path_cfg.images_dir, path_cfg.smplx_gts, path_cfg.csv_file
    )
    transform = build_transforms()
    trainset, testset = get_datasets(path_cfg.csv_file, transform, test_size=0.1)
    trainloader, valloader = get_dataloader(
        trainset, train_cfg.batch_size, type="train", val_size=0.2
    )
    testloader = get_dataloader(testset, train_cfg.batch_size, type="test")

    wandb_logger = wandb_logger = WandbLogger(
        name=train_cfg.experiment_name,
        project=train_cfg.project_name,
        log_model="all",
        # offline=True,
        save_dir=path_cfg.log_dir,
    )
    # path_cfg.checkpoint_file = "img2smplx_2023-11-30_13-11-21.ckpt"
    model, result = train_model(
        path_cfg,
        data_cfg,
        train_cfg,
        trainloader,
        valloader,
        testloader,
        False,
        "train",
        wandb_logger,
    )

    print(result)


if __name__ == "__main__":
    main()

from utils.initialize import device

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models import Img2SMPLx


def train_model(
    path_cfg,
    data_cfg,
    train_cfg,
    trainloader,
    valloader,
    testloader,
    load_ckpt=False,
    mode="train",
    logger=None,
):
    trainer = pl.Trainer(
        default_root_dir=path_cfg.checkpoint_dir,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=train_cfg.epochs,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=path_cfg.checkpoint_dir,
                filename=path_cfg.checkpoint_file,
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=logger,
    )
    trainer.logger._log_graph = True

    pretrained_model = os.path.join(path_cfg.checkpoint_dir, path_cfg.checkpoint_file)
    # Check if pretrained model already exists
    if load_ckpt:
        if os.path.isfile(pretrained_model):
            print(f"Found saved model checkpoint at {pretrained_model}, loading...")
            model = Img2SMPLx.load_from_checkpoint(
                pretrained_model,
                path_cfg=path_cfg,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
            )
        else:
            raise FileNotFoundError(f"Model Checkpoint at {pretrained_model} not found")
    if mode == "train":
        model = Img2SMPLx(path_cfg, data_cfg, train_cfg)
        print(model)
        trainer.fit(model, trainloader, valloader)
        # Loading the best model after training
        model = Img2SMPLx.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Testing best model on val and test sets
    val_result = trainer.validate(model, valloader)
    test_result = trainer.test(model, testloader)
    result = {"final_test_loss": test_result, "final_val_loss": val_result}
    # if logger:
    #     logger.log(result)

    return model, result

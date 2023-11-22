from utils.consts import EPOCHS, CHECKPOINT_PATH
from utils.consts import device

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models import Img2SMPLx


def train_model(args, trainloader, valloader, testloader, logger):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "img2smplx"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=EPOCHS,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_loss"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=logger,
    )
    trainer.logger._log_graph = True

    # Check if pretrained model already exists
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "img2smplx.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found saved model checkpoint at {pretrained_filename}, loading...")
        model = Img2SMPLx.load_from_checkpoint(pretrained_filename)
    else:
        model = Img2SMPLx(*args)
        trainer.fit(model, trainloader, valloader)
        # Loading the best model after training
        model = Img2SMPLx.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Testing best model on val and test sets
    val_result = trainer.validate(model, valloader)
    test_result = trainer.test(model, testloader)

    result = {"test": test_result, "val": val_result}

    return model, result

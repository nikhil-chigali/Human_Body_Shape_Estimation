import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from models import VisionTransformer
import smplx
from utils.consts import device


class Img2SMPLx(pl.LightningModule):
    def __init__(
        self, model_kwargs, smplx_path, batch_size, num_betas, lr, lr_milestones
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vit = VisionTransformer(**model_kwargs)
        self.smplx_male = smplx.create(
            model_path=smplx_path,
            gender="male",
            num_betas=num_betas,
            batch_size=batch_size,
            model_type="smplx",
        )
        self.smplx_female = smplx.create(
            model_path=smplx_path,
            gender="female",
            num_betas=num_betas,
            batch_size=batch_size,
            model_type="smplx",
        )
        self.smplx_neutral = smplx.create(
            model_path=smplx_path,
            gender="neutral",
            num_betas=num_betas,
            batch_size=batch_size,
            model_type="smplx",
        )

    def make_gender_mask(self, genders):
        ones = torch.ones(size=(10475, 3))
        zeros = torch.zeros(size=(10475, 3))
        encoding = {
            "male": torch.stack([zeros, zeros, ones]),
            "female": torch.stack([zeros, ones, zeros]),
            "neutral": torch.stack([ones, zeros, zeros]),
        }
        self.gender_mask = torch.stack(
            [encoding[gender] for gender in genders]
        )  # [B, 3, 10475, 3]
        self.gender_mask = self.gender_mask.to(device)

    def forward(self, x, genders):
        self.betas = self.vit(x, genders)
        vertices = torch.stack(
            [
                self.smplx_neutral(betas=self.betas).vertices,
                self.smplx_female(betas=self.betas).vertices,
                self.smplx_male(betas=self.betas).vertices,
            ]
        ).transpose(0, 1)
        self.make_gender_mask(genders)
        out = (vertices * self.gender_mask).sum(axis=1)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.vit.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.lr_milestones,
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        X, y = batch["X"], batch["y"]
        imgs, genders, verts = X[0].to(device), X[1], y.to(device)
        preds = self.forward(imgs, genders)
        loss = F.mse_loss(preds, verts)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

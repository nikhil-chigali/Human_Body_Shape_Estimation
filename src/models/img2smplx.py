import torch
import torch.optim as optim

import pytorch_lightning as pl

from models import VisionTransformer
import smplx
from utils.initialize import device


class Img2SMPLx(pl.LightningModule):
    def __init__(self, path_cfg, data_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.lr = train_cfg.lr
        self.lr_milestones = train_cfg.lr_milestones
        model_kwargs = {
            "segment_size": data_cfg.strip_thickness,
            "num_layers": data_cfg.num_layers,
            "embed_size": data_cfg.embed_size,
            "hidden_size": data_cfg.hidden_size,
            "num_heads": data_cfg.num_heads,
            "img_size": data_cfg.img_size,
            "num_channels": 3,
            "output_size": data_cfg.num_betas,
            "segment_type": data_cfg.img_segment_type,
            "dropout": train_cfg.dropout,
        }
        self.model = VisionTransformer(**model_kwargs)
        self.smplx_male = smplx.create(
            model_path=path_cfg.smplx_model_dir,
            gender="male",
            num_betas=data_cfg.num_betas,
            batch_size=train_cfg.batch_size,
            model_type="smplx",
        )
        self.smplx_female = smplx.create(
            model_path=path_cfg.smplx_model_dir,
            gender="female",
            num_betas=data_cfg.num_betas,
            batch_size=train_cfg.batch_size,
            model_type="smplx",
        )
        self.smplx_neutral = smplx.create(
            model_path=path_cfg.smplx_model_dir,
            gender="neutral",
            num_betas=data_cfg.num_betas,
            batch_size=train_cfg.batch_size,
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

    def point_error(
        self,
        x,
        y,
        align: bool = True,
    ) -> float:
        t = 0.0
        if align:
            t = x.mean(0, keepdims=True) - y.mean(0, keepdims=True)

        x_hat = x - t

        error = torch.pow(torch.pow(x_hat - y, 2).sum(axis=-1), 0.5)

        return error.mean()

    def forward(self, x, genders):
        betas = self.model(x, genders)
        vertices = torch.stack(
            [
                self.smplx_neutral(betas=betas).vertices,
                self.smplx_female(betas=betas).vertices,
                self.smplx_male(betas=betas).vertices,
            ]
        ).transpose(0, 1)
        self.make_gender_mask(genders)
        out = (vertices * self.gender_mask).sum(axis=1)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.lr_milestones,
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def p2p_loss_10k(self, pred_points, gt_points):
        # Step 1: Randomly select 10,000 pairs of points
        indices = torch.randperm(pred_points.size(0))[:10000]
        pred_points = pred_points[indices]
        gt_points = gt_points[indices]

        # Step 2: Compute Euclidean distance
        distance = torch.norm(pred_points - gt_points, dim=1)

        # Step 3: Average distance
        loss = torch.mean(distance)

        return loss

    def p2p_loss_20k(self, pred_points, gt_points):
        # Step 1: Randomly select 20,000 pairs of points
        indices = torch.randperm(pred_points.size(0))[:20000]
        pred_points = pred_points[indices]
        gt_points = gt_points[indices]

        # Step 2: Compute Euclidean distance
        distance = torch.norm(pred_points - gt_points, dim=1)

        # Step 3: Average distance
        loss = torch.mean(distance)

        return loss

    def _calculate_loss(self, batch, mode="train"):
        X, y = batch["X"], batch["y"]
        imgs, genders, verts = X[0].to(device), X[1], y.to(device)
        preds = self.forward(imgs, genders)
        # print(
        #     self.p2p_loss_10k(preds, verts).item(),
        #     self.p2p_loss_20k(preds, verts).item(),
        #     self.point_error(preds, verts).item(),
        # )
        # loss = F.l1_loss(preds, verts) + self.point_error(preds, verts)
        loss = self.p2p_loss_10k(preds, verts) + self.p2p_loss_20k(preds, verts)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

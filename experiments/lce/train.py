"""
Definition of the model
"""

import torch
from torch import nn
import wandb
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from lidar_confidence.dataset import Dataset
from lidar_confidence.metrics import cost_curve_aucs
from lidar_confidence.image import normalize, color_depth
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import multiprocessing
from os import path
from pathlib import Path
from typing import Literal, Optional
import yaml
import json
import shutil
from model import LCE

__all__ = ["LCETrain", "DataModule"]

# % DATA LOAD AND TRANSFORM
# this transformations are applied to each batch in order to perform data
# augmentation or simply adaptation of each example


def preprocess(d: dict) -> dict:

    # convert into tensors and crop
    for k in d:
        d[k] = F.to_tensor(d[k])

    # normalize image
    d["img"] = F.normalize(
        d["img"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return d


class DataModule(LightningDataModule):
    def __init__(self, root: str, batch_size: int = 1, **kwargs):
        super().__init__()

        self.root = root
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_ds = Dataset(self.root, split="train", transform=preprocess)
        self.val_ds = Dataset(self.root, split="val", transform=preprocess)
        self.test_1_ds = Dataset(self.root, split="test_1", transform=preprocess)
        self.test_2_ds = Dataset(self.root, split="test_2", transform=preprocess)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )

    def test_dataloader(self):
        test_1 = DataLoader(
            self.test_1_ds,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )

        test_2 = DataLoader(
            self.test_2_ds,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )

        return [test_1, test_2]


# % TRAIN PROCEDURE
# here there is the training procedure wrapped into a LightningModule
# subclass, the key point is that we train the model provinding in input
# the image and the lidar output and we constraint the network to reproduce
# the input but inducing it to generate also the standard deviation of
# its measures.


class SparseMinPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(9, 1, 4)

    def forward(self, x):
        x_max = x.max()
        x = torch.where(
            x > 0, x_max - x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )
        x = self.max_pool(x)
        x = torch.where(
            x > 0, x_max - x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )
        return x


class SparseAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum_pool = nn.AvgPool2d(9, 1, 4, divisor_override=1)

    def forward(self, x):
        num_pool = self.sum_pool((x > 0).to(x.dtype))
        avg_pool = self.sum_pool(x)
        avg_pool[num_pool > 0] = avg_pool[num_pool > 0] / num_pool[num_pool > 0]
        return avg_pool


class LCETrain(LightningModule):

    # model

    def __init__(
        self,
        lr: float = 1e-4,
        head: Literal["mlp", "decoder"] = "mlp",
        depth_supervision: Literal["min_pooling", "avg_pooling", "gt"] = "min_pool",
        loss_fn: Literal["l1", "l2", "l2+1", "l2+1norm"] = "l2+1norm",
        **kwargs,
    ):
        super().__init__()
        assert head in ["mlp", "decoder"]
        assert depth_supervision in ["min_pooling", "avg_pooling", "gt"]
        assert loss_fn in ["l1", "l2", "l2+1", "l2+1norm"]

        self.loss_fn = loss_fn

        self._val_step = 0
        self._test_1_step = 0
        self._test_2_step = 0

        self.model = LCE(head=head)

        if depth_supervision == "min_pooling":
            self.model_depth = SparseMinPooling()
        else:
            self.model_depth = SparseAvgPooling()

        self.hparams.num_params = sum(p.numel() for p in self.model.parameters())
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    # optimizer

    def configure_optimizers(self):
        std_opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return std_opt

    # training & val procedure

    def _compute_input(self, batch):
        input = torch.cat([batch["img"], batch["lidar"]], 1)
        return input

    def _compute_supervision(self, batch):

        if self.hparams.depth_supervision != "gt":
            y = batch["lidar"]
        else:
            y = torch.where(
                (batch["lidar"] > 0) & (batch["gt"] > 0),
                batch["lidar"],
                torch.tensor(
                    0.0, dtype=batch["lidar"].dtype, device=batch["lidar"].device
                ),
            )

        return y

    def _compute_std_loss(self, batch):

        X = self._compute_input(batch)

        y = self._compute_supervision(batch)
        mask = y > 0
        y = y[mask]

        if self.hparams.depth_supervision != "gt":
            depth_pred = self.model_depth(X[:, -1:])
        else:
            depth_pred = batch["gt"]

        inv_conf_pred = self.model(X)

        mu = depth_pred[mask]
        sigma = inv_conf_pred[mask] + 1

        if "norm" in self.loss_fn:
            diff = torch.abs(mu - y) / (mu + 1)
        else:
            diff = torch.abs(mu - y)

        if "+1" in self.loss_fn:
            diff = diff + 1

        if "l1" in self.loss_fn:
            err = torch.mean(torch.abs(diff / sigma) + torch.log(sigma))
        else:
            err = torch.mean(torch.square(diff / sigma) + torch.log(sigma))

        return err, depth_pred, inv_conf_pred

    def _log_metrics(self, batch, loss, out, stage):

        if stage == "train":
            on_step, on_epoch = True, False
        else:
            on_step, on_epoch = False, True

        # loss logging
        self.log(f"loss/{stage}", loss, on_epoch=on_epoch, on_step=on_step)

        # depth logging
        y = self._compute_supervision(batch)
        mask = y > 0
        depth = out[:, 0:1]
        rmse = torch.sqrt(torch.mean(torch.square(depth[mask] - y[mask])))
        self.log(f"depth_rmse/{stage}", rmse, on_epoch=on_epoch, on_step=on_step)

        # auc logging
        std = out[:, 1:2]
        confs = torch.zeros_like(std)
        confs = std.max() - std

        for crit in ["mae", "rmse"]:
            auc, auc_opt, auc_rnd = cost_curve_aucs(
                batch["gt"], batch["lidar"], confs, crit
            )

            self.log_dict(
                {
                    f"auc_{crit}/{stage}": auc,
                    f"auc_opt_{crit}/{stage}": auc_opt,
                    f"auc_rnd_{crit}/{stage}": auc_rnd,
                },
                on_epoch=on_epoch,
                on_step=on_step,
            )

        # images logging
        global_step = self.global_step
        if (
            (stage == "train" and global_step % 1000 == 0)
            or (stage == "val" and self._val_step % 1000 == 0)
            or (stage == "test_1" and self._test_1_step % 100 == 0)
            or (stage == "test_2" and self._test_2_step % 10 == 0)
        ):

            image = normalize(batch["img"][0]).permute(1, 2, 0).cpu().numpy()
            lidar = (
                color_depth(batch["lidar"][0], invert=True)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            gt = color_depth(batch["gt"][0], invert=True).permute(1, 2, 0).cpu().numpy()
            confidence = color_depth(std[0].detach()).permute(1, 2, 0).cpu().numpy()

            depth = depth[0].max() - depth[0]
            depth_img = (
                color_depth(depth.detach(), cmap="viridis_r")
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            self.logger.experiment.log(
                {
                    f"media/{stage}": [
                        wandb.Image(image, caption="image"),
                        wandb.Image(lidar, caption="lidar"),
                        wandb.Image(gt, caption="groundtruth"),
                        wandb.Image(confidence, caption="confidence"),
                        wandb.Image(depth_img, caption="depth"),
                    ]
                },
            )

    def training_step(self, batch, batch_idx):

        loss, mu, std = self._compute_std_loss(batch)
        out = torch.cat([mu, std], 1)
        self._log_metrics(batch, loss, out, "train")

        return loss

    def on_validation_start(self):
        self._val_step = 0

    def validation_step(self, batch, batch_idx):
        self._val_step += 1
        loss, mu, std = self._compute_std_loss(batch)
        out = torch.cat([mu, std], 1)
        self._log_metrics(batch, loss, out, "val")
        return loss

    def on_test_start(self):
        self._test_1_step = 0
        self._test_2_step = 0

    def test_step(self, batch, batch_idx, dl_idx):
        if dl_idx == 0:
            self._test_1_step += 1
        if dl_idx == 1:
            self._test_2_step += 1

        loss, mu, std = self._compute_std_loss(batch)
        out = torch.cat([mu, std], 1)

        self._current_dataloader_idx = None  # (disable idx appending)
        self._log_metrics(batch, loss, out, f"test_{dl_idx + 1}")
        return loss


# % TRAINING SCRIPT

if __name__ == "__main__":

    pl.seed_everything(10)

    # load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["lce"]

    # setup results path
    results_path = Path("results") / "lce"
    results_path.mkdir(parents=True, exist_ok=True)

    # load data and model
    data = DataModule(path.join("data", "dataset"), **params)
    data.prepare_data()
    data.setup()

    model = LCETrain(**params)

    # ckpt setup
    ckpt_path = results_path / "ckpts"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_path.as_posix(),
        monitor="auc_mae/val",
    )

    # logger setup
    wandb_logger = WandbLogger(log_model=False)

    # trainer setup
    trainer = Trainer(
        weights_summary=None,
        max_epochs=params["epochs"],
        precision=16,
        gpus=1,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
    )

    # train
    trainer.fit(model, data)

    # test
    metrics = trainer.test(
        model,
        ckpt_path=ckpt_callback.best_model_path,
        datamodule=data,
        verbose=False,
    )

    # save metrics in results folder
    to_save = {}
    for m in metrics:
        for k, v in m.items():
            to_save["/".join(k.split("/")[::-1])] = round(v, 7)

    with open(results_path / "lce.json", "w") as f:
        json.dump(to_save, f)

    # save wandb info in results folder
    to_save = {
        "name": wandb_logger.experiment.name,
        "id": wandb_logger.experiment.id,
    }
    with open(results_path / "wandb_info.json", "w") as f:
        json.dump(to_save, f)

    # save wandb training code for reference
    final_dir = Path(wandb_logger.experiment.dir) / "train_code"
    final_dir.mkdir(exist_ok=True)
    model_file = final_dir / "model.py"
    train_file = final_dir / "train.py"

    shutil.copyfile(Path(__file__).parent / "model.py", model_file)
    shutil.copyfile(__file__, train_file)

    # save the best model and delete the others
    best_path = ckpt_callback.best_model_path
    model = LCETrain.load_from_checkpoint(best_path, map_location=torch.device("cpu"))
    model = model.model
    model.filter_output = True
    model_script = torch.jit.script(model)
    torch.jit.save(model_script, (results_path / "model.pth").as_posix())
    shutil.rmtree(ckpt_path.as_posix(), ignore_errors=True)

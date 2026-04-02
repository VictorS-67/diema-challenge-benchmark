"""Model-agnostic Lightning training wrapper.

Works with any model that follows the BaseModel protocol (returns a dict
with at least {"logits": tensor}). Handles auxiliary losses generically
via the "aux_losses" key.
"""

import torch
import torch.nn as nn

import torchmetrics
import pytorch_lightning as pl


class LightningModel(pl.LightningModule):
    """Lightning wrapper for training any BaseModel subclass.

    Args:
        model: a BaseModel instance
        base_lr: base learning rate
        num_class: number of classes for metrics
        optimizer: 'SGD' or 'Adam' (default: 'SGD')
        scheduler_type: 'cosine' or 'step' (default: 'cosine')
        scheduler_params: for 'step', provide [step_size, gamma] (default: [])
        weight_decay: L2 regularization (default: 0.0001)
        aux_loss_weights: per-loss weight overrides, e.g. {"ab_logits": 0.5} (default: all 1.0)
    """

    def __init__(
        self,
        model,
        base_lr,
        num_class,
        optimizer="SGD",
        scheduler_type="cosine",
        scheduler_params=None,
        weight_decay=0.0001,
        aux_loss_weights=None,
    ):
        super().__init__()
        if scheduler_params is None:
            scheduler_params = []

        self.base_lr = base_lr
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.weight_decay = weight_decay
        self.aux_loss_weights = aux_loss_weights or {}

        if scheduler_type not in ("cosine", "step"):
            raise ValueError("Unsupported scheduler type")
        if optimizer not in ("SGD", "Adam"):
            raise ValueError("Unsupported optimizer type")
        if scheduler_type == "step" and not scheduler_params:
            raise ValueError("Scheduler params must be provided for step scheduler")

        self.save_hyperparameters(ignore=["model"])

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_class, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)  # dict with at least {"logits": ...}
        loss = self.loss_fn(out["logits"], labels)

        # Handle auxiliary losses if the model produces them
        if "aux_losses" in out:
            for name, value in out["aux_losses"].items():
                if name.endswith("_logits"):
                    aux_loss = self.loss_fn(value, labels)
                else:
                    aux_loss = value
                weight = self.aux_loss_weights.get(name, 1.0)
                loss = loss + weight * aux_loss
                self.log(f"train_{name}", aux_loss)

        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        loss = self.loss_fn(out["logits"], labels)
        predicted = torch.argmax(out["logits"], dim=1)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))
        self.val_acc(predicted, labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        predicted = torch.argmax(out["logits"], dim=1)

        self.test_acc(predicted, labels)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.test_f1(predicted, labels)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inputs, labels, sample_names = batch
        out = self(inputs)
        proba = torch.softmax(out["logits"], dim=1)
        predicted = torch.argmax(proba, dim=1)
        return predicted, proba, labels, sample_names

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )

        if self.scheduler_type == "step":
            step_size, gamma = self.scheduler_params
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

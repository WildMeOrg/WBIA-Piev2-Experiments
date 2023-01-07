from argparse import ArgumentParser
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from src.metrics import (
    avg_pair_score_distances,
    get_topk_acc,
    validation_stats,
)
from src.loss import PIEv2Loss
from torch.optim.lr_scheduler import StepLR

from transformers import ViTFeatureExtractor, ViTModel


class ViTEmbeddingModule(pl.LightningModule):
    def __init__(self, margin, num_classes, weight_t, weight_x, embedding_dim, lr, wd, fixbase_epoch, name, **kwargs):
        super().__init__()

        self.embedding_dim = embedding_dim

        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        self.model = model = ViTModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.pooler.dense.in_features, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.lr = lr
        self.wd = wd

        self.fixbase_epoch = fixbase_epoch

        self.name = name

        self.save_hyperparameters()

        self.criterion = PIEv2Loss(
            margin, self.embedding_dim, num_classes, False, weight_t, weight_x
        )

        self.topil = T.ToPILImage()

    def forward(self, batch):
        x, _, annot, names = batch
        x = [self.topil(img) for img in x]
        x = self.feature_extractor(x, return_tensors='pt')
        x = x['pixel_values'].to(y.device)
        x = self.model(x).pooler_output
        x = self.classifier(x)

        return {"embeddings": x, "names": names, "annots": annot}

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        x = [self.topil(img) for img in x]
        x = self.feature_extractor(x, return_tensors='pt')
        x = x['pixel_values'].to(y.device)
        x = self.model(x).pooler_output
        x = self.classifier(x)

        simmat = self._get_simmat(x)
        pos, neg = avg_pair_score_distances(simmat, y)

        loss = self.criterion(x, y)

        self.log("train_pos_score", pos)
        self.log("train_neg_score", neg)
        self.log("train_dist", pos - neg, prog_bar=True)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        x = [self.topil(img) for img in x]
        x = self.feature_extractor(x, return_tensors='pt')
        x = x['pixel_values'].to(y.device)
        x = self.model(x).pooler_output
        x = self.classifier(x)

        # We cannot compute loss on the validation set since it uses different individuals
        loss = torch.tensor(-1.0)

        return {"embeddings": x, "labels": y, "loss": loss}

    def validation_epoch_end(self, val_outs):
        batch_embeddings = [out["embeddings"] for out in val_outs]
        batch_labels = [out["labels"] for out in val_outs]
        batch_losses = [out["loss"].view(1, 1) for out in val_outs]
        embeddings = torch.cat(batch_embeddings)
        labels = torch.cat(batch_labels)
        losses = torch.cat(batch_losses)

        loss = losses.mean()

        simmat = self._get_simmat(embeddings)

        pos, neg = avg_pair_score_distances(simmat, labels)
        self.log("val_pos_score", pos)
        self.log("val_neg_score", neg)
        self.log("val_dist", pos - neg)
        self.log("val_loss", loss, prog_bar=True)

        (top1, top5, top10), m_a_p = get_topk_acc(simmat, labels)
        roc_auc, pr_auc = validation_stats(simmat, labels)

        self.log("top1", top1, prog_bar=True)
        self.log("top5", top5)
        self.log("top10", top10)
        self.log("mAP@R", m_a_p)
        self.log("roc_auc", roc_auc)
        self.log("pr_auc", pr_auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, _, y = batch

        x = [self.topil(img) for img in x]
        x = self.feature_extractor(x, return_tensors='pt')
        x = x['pixel_values'].to(y.device)
        x = self.model(x).pooler_output
        x = self.classifier(x)

        return {"embeddings": x, "labels": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, 120),
        }

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="top1",
                mode="max",
                filename="{epoch}-{top1:.2f}",
                every_n_epochs=2,
            ),
            FixBase(self.fixbase_epoch),
        ]

    def _get_simmat(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        distmat = self.get_distmat(embeddings, embeddings)

        # [0, rad2]
        simmat = 1 - distmat / math.sqrt(2)

        return simmat
    
    @staticmethod
    def get_distmat(emb1, emb2):
        m, n = emb1.size(0), emb2.size(0)
        mat1 = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
        distmat = distmat.clamp(min=1e-12).sqrt()

        return distmat

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, allow_abbrev=False
        )
        parser.add_argument("--embedding-dim", default=512, type=int)
        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--wd", default=5e-4, type=float)
        parser.add_argument("--fixbase-epoch", default=1, type=int)
        parser.add_argument("--margin", default=0.3, type=float)
        parser.add_argument("--weight-t", default=1.0, type=float)
        parser.add_argument("--weight-x", default=1.0, type=float)

        return parser


class FixBase(Callback):
    def __init__(self, fixbase_epoch):
        self.fixbase_epoch = fixbase_epoch

    def on_train_start(self, trainer, pl_module):
        if self.fixbase_epoch != 0 and pl_module.current_epoch == 0:
            print("\nFreezing backbone weights excluding fc...")
            for name, param in pl_module.model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

    def on_train_epoch_end(self, trainer, pl_module):
        if (self.fixbase_epoch - 1) == pl_module.current_epoch:
            print("\nUnfreezing backbone weights excluding fc...")
            for name, param in pl_module.model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = True

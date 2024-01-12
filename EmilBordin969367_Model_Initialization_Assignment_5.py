import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models.detection
from pytorch_lightning import LightningModule
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class FasterRCNN(LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = 1e-4
        self.batch_size = 8

    def forward(self, imgs):

        return self.detector(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        targets = []

        for boxes,labels0 in zip(batch[1], batch[2]):
            target = {}
            target["boxes"] = boxes
            labels = torch.flatten(labels0)
            target["labels"] = labels
            targets.append(target)

        loss_dict = self.detector(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, batch_size=self.batch_size)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        img, boxes, label = batch
        predictions = self.forward(img)
        val_accuracy = torch.mean(
            torch.stack([self.mean_average_precision(b,l,p)["map_75"] for b,l,p, in zip(boxes, label, predictions)]))
        self.log("val_accuracy", val_accuracy, batch_size=self.batch_size)
        return val_accuracy

    def test_step(self, batch, batch_idx):
        img, boxes, label = batch
        predictions = self.forward(img)
        test_accuracy = torch.mean(
            torch.stack([self.mean_average_precision(b,l,p)["map_75"] for b,l,p in zip(boxes, label, predictions)]))
        self.log("test_accuracy", test_accuracy, batch_size=self.batch_size)
        return test_accuracy

    def mean_average_precision(self, src_boxes: list, src_labels: list, predictions: dict) -> dict:
        """
        Calculation of different mean average precision and mean average recall metrics:
        https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html

        Parameters:
        src_boxes: List of true bounding boxes per image
        src_labels: List of true bounding box labels per image
        predictions: Dictionary of predictions containing bounding box, label and score


        Returns:
        metric_dict: Dictionary containing different mean average precision and mean average recall metrics

        """

        metric = MeanAveragePrecision()
        target = dict(
            boxes=src_boxes,
            labels=torch.flatten(src_labels))
        metric.update([predictions], [target])
        metric_dict = metric.compute()

        return metric_dict

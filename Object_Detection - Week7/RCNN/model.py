import torch
import torch.nn as nn
from torchvision import models

from train import label2target


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 25088
        
        vgg_backbone = models.vgg16(pretrained=True)
        vgg_backbone.classifier = nn.Sequential()
        for param in vgg_backbone.parameters():
            param.requires_grad = False
        
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
    def forward(self, input):
        scores = torch.zeros(input.shape[0], 2).cuda()
        bboxes = torch.zeros(input.shape[0],4).cuda()
        for i in range(0,len(input), 32):
          inputs = input[i:i+32]
          feat = self.backbone(inputs)
          cls_score = self.cls_score(feat)
          bbox = self.bbox(feat)

          scores[i:i+32] = cls_score
          bboxes[i:i+32] = bbox
        return scores, bboxes
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != label2target['background'])
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss
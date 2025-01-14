import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def makd_loss(logits_student_aspect, logits_makd_aspect):
    loss_makd = F.binary_cross_entropy_with_logits(logits_student_aspect, logits_makd_aspect, reduction='mean')
    return loss_makd

class MaKD(Distiller):
    """MULTI-ASPECT KNOWLEDGE DISTILLATION WITH LARGE LANGUAGE MODEL"""

    def __init__(self, student, cfg):
        super(MaKD, self).__init__(student, cfg)
        self.ce_loss_weight = cfg.MaKD.CE_WEIGHT
        self.alpha = cfg.MaKD.ALPHA
        self.temperature = cfg.MaKD.TEMPERATURE
        self.aspect_num = cfg.MaKD.ASPECT_NUM
        self.num_classes = cfg.DATASET.NUM_CLASS

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def forward_train(self, image, target, **kwargs):
        makd_logits = list(kwargs.items())[1][1]
        logits_student, _ = self.student(image)
        logits_student_aspect = logits_student[:, self.num_classes:self.num_classes + self.aspect_num]
        logits_student = logits_student[:, :self.num_classes]
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_makd = self.alpha * makd_loss(logits_student_aspect, makd_logits)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_makd": loss_makd,
        }
        return logits_student, losses_dict



import random
import os
import sys
from pathlib import Path

current_path = Path(__file__).resolve()
parent_folder = current_path.parent.parent
sys.path.append(str(parent_folder))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from mdistiller.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, finegrained
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
import subprocess
import wandb
import argparse

def set_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(cfg, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            #wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    show_cfg(cfg)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize, ])
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ])

    train_set = finegrained.ImageDataset(json_path=cfg.DATASET.ASPECT_JSON, image_folder=cfg.DATASET.TRAIN_IMAGE_PATH, transform=train_transform)
    test_set = finegrained.TestImageDataset(image_folder=cfg.DATASET.TEST_IMAGE_PATH, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                              num_workers=1, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=cfg.DATASET.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=1, drop_last=True)

    class_num = cfg.DATASET.NUM_CLASS
    aspect_num = cfg.MaKD.ASPECT_NUM
    model_output_num = class_num + aspect_num
    model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, num_classes=model_output_num, )

    distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, cfg
            )

    distiller = torch.nn.DataParallel(distiller.cuda())

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.opts)

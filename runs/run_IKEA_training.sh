#!/bin/bash

python -m tadn.online_training \
    dataset/train_transforms=basic_app_transforms \
    dataset/val_transforms=basic_app_transforms \
    dataset.root='/data/IKEA_MOT_dataset' \
    trainer.workdir='./runs/outputs'
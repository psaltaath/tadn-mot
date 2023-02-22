from typing import Callable, Tuple
import pandas as pd
import numpy as np
from functools import lru_cache
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .base import MOTDataset, OnlineTrainingDatasetWrapper
from .mot_challenge import MOTChallengeDataset, MOTChallengeCategorySet
from .detrac import DetracDataset
from .transforms import AVAILABLE_TRANSFORMS


def build_datasets(
    dataset_cfg: DictConfig,
) -> Tuple[OnlineTrainingDatasetWrapper, OnlineTrainingDatasetWrapper]:
    train_T = [
        AVAILABLE_TRANSFORMS[T.fun](**T.kwargs) for T in dataset_cfg.train_transforms
    ]
    val_T = [
        AVAILABLE_TRANSFORMS[T.fun](**T.kwargs) for T in dataset_cfg.val_transforms
    ]

    if dataset_cfg.type == "mot17":
        if dataset_cfg.split in ["default", "none"]:
            train_mode = "train"
            val_mode = "train"
        elif dataset_cfg.split == "half":
            train_mode = "train_50"
            val_mode = "val_50"
        else:
            raise NotImplementedError

        train_dset = MOTChallengeDataset(
            category_set=MOTChallengeCategorySet.TRAINING,
            detector=dataset_cfg.detector,
            mode=train_mode,
            version=dataset_cfg.type.upper(),
            root=dataset_cfg.root,
            ignore_MOTC=False,
            load_frame_data=False,
            transforms=train_T,
        )

        val_dset = MOTChallengeDataset(
            category_set=MOTChallengeCategorySet.TRAINING,
            detector=dataset_cfg.detector,
            mode=val_mode,
            version=dataset_cfg.type.upper(),
            root=dataset_cfg.root,
            ignore_MOTC=False,
            load_frame_data=False,
            transforms=val_T,
        )

    elif dataset_cfg.type == "detrac":
        train_dset = DetracDataset(
            detector=dataset_cfg.detector,
            mode="train",
            root=dataset_cfg.root,
            ignore_MOTC=False,
            load_frame_data=False,
            transforms=train_T,
        )

        val_dset = DetracDataset(
            detector=dataset_cfg.detector,
            mode="test",
            root=dataset_cfg.root,
            ignore_MOTC=False,
            load_frame_data=False,
            transforms=val_T,
        )
    else:
        raise AssertionError("Invalid dataset type")

    train_online_dset = OnlineTrainingDatasetWrapper(
        train_dset, skip_first_frame=dataset_cfg.skip_first_frame
    )

    val_online_dset = OnlineTrainingDatasetWrapper(
        val_dset, skip_first_frame=dataset_cfg.skip_first_frame
    )

    return train_online_dset, val_online_dset


def build_dataloaders(train_dset, val_dset, dataloader_cfg: DictConfig) -> Tuple:
    train_dloader = DataLoader(
        train_dset,
        num_workers=dataloader_cfg.num_workers,
        prefetch_factor=dataloader_cfg.prefetch_factor,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    val_dloader = DataLoader(
        val_dset,
        num_workers=dataloader_cfg.num_workers,
        prefetch_factor=dataloader_cfg.prefetch_factor,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_dloader, val_dloader

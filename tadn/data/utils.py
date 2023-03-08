from typing import Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .base import OnlineTrainingDatasetWrapper
from .mot_challenge import MOTChallengeDataset, MOTChallengeCategorySet
from .detrac import DetracDataset
from .transforms import AVAILABLE_TRANSFORMS


def build_datasets(
    dataset_cfg: DictConfig, skip_train: bool = False, skip_val: bool = False
) -> Tuple[OnlineTrainingDatasetWrapper, OnlineTrainingDatasetWrapper]:
    if not skip_train:
        train_T = [
            AVAILABLE_TRANSFORMS[T.fun](**T.kwargs)
            for T in dataset_cfg.train_transforms
        ]
    if not skip_val:
        val_T = [
            AVAILABLE_TRANSFORMS[T.fun](**T.kwargs) for T in dataset_cfg.val_transforms
        ]

    if dataset_cfg.type == "mot17":
        if dataset_cfg.split == "default":
            train_mode = "train"
            val_mode = "test"
        elif dataset_cfg.split == "only_train":
            train_mode = "train"
            val_mode = "train"
        elif dataset_cfg.split == "half":
            train_mode = "train_50"
            val_mode = "val_50"
        else:
            raise NotImplementedError

        if not skip_train:
            train_dset = MOTChallengeDataset(
                category_set=MOTChallengeCategorySet.TRAINING,
                detector=dataset_cfg.detector,
                mode=train_mode,
                version=dataset_cfg.type.upper(),
                root=dataset_cfg.root,
                ignore_MOTC=False,
                load_frame_data=False,
                transforms=train_T,  # type: ignore
            )

        if not skip_val:
            val_dset = MOTChallengeDataset(
                category_set=MOTChallengeCategorySet.TRAINING,
                detector=dataset_cfg.detector,
                mode=val_mode,
                version=dataset_cfg.type.upper(),
                root=dataset_cfg.root,
                ignore_MOTC=False,
                load_frame_data=False,
                transforms=val_T,  # type: ignore
            )

    elif dataset_cfg.type == "detrac":
        if not skip_train:
            train_dset = DetracDataset(
                detector=dataset_cfg.detector,
                mode="train",
                root=dataset_cfg.root,
                ignore_MOTC=False,
                load_frame_data=False,
                transforms=train_T,  # type: ignore
            )

        if not skip_val:
            val_dset = DetracDataset(
                detector=dataset_cfg.detector,
                mode="test",
                root=dataset_cfg.root,
                ignore_MOTC=False,
                load_frame_data=False,
                transforms=val_T,  # type: ignore
            )
    else:
        raise AssertionError("Invalid dataset type")

    outputs = []
    if not skip_train:
        train_online_dset = OnlineTrainingDatasetWrapper(
            train_dset, skip_first_frame=dataset_cfg.skip_first_frame  # type: ignore
        )
        outputs.append(train_online_dset)

    if not skip_val:
        val_online_dset = OnlineTrainingDatasetWrapper(
            val_dset, skip_first_frame=dataset_cfg.skip_first_frame  # type: ignore
        )
        outputs.append(val_online_dset)

    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def build_dataloaders(*dsets, dataloader_cfg: DictConfig) -> Tuple:
    outputs = []
    for dset in dsets:
        outputs.append(
            DataLoader(
                dset,
                num_workers=dataloader_cfg.num_workers,
                prefetch_factor=dataloader_cfg.prefetch_factor,
                batch_size=dataloader_cfg.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        )

    return tuple(outputs) if len(outputs) > 1 else outputs[0]

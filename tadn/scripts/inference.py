from argparse import ArgumentParser
from typing import Any, Dict, Iterable

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from config4ml.lightning.extra import ConsoleLogger

from src.config.data import MOTChallengeDatasetConfig, MOTDatasetConfig
from src.data.base import OnlineTrainingDatasetWrapper
from src.data.mot_challenge import MOTChallengeCategorySet, MOTChallengeDataset

from ..components import tracklets
from ..config.experiment import ExperimentConfig
from ..online_training import OnlineManager, OnlineTraining, init_model_from_config
from ..components.transformer import TADN
from ..utils.bbox import convert_MOTC_format
from ..utils.tracklets import truncate_tracklets_MOTC_format
from ..data.transforms import (
    load_app_vectors,
    filter_det,
    normalize,
    load_ecc_transforms,
)


def load_from_ckpt(ckpt_file, cfg_file):
    cfg = ExperimentConfig.parse_file(cfg_file)

    model = init_model_from_config(cfg)

    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["state_dict"])

    return model


# def load_dloaders():
#     train_dset = MOTChallengeDataset(
#         type="mot-challenge",
#         root="/data/MOT17",
#         mode="train",
#         transforms=[
#             load_app_vectors(
#                 data_root="/data/MOT17", mode="train", feature_extractor="reid"
#             ),
#             load_ecc_transforms(data_root="/data/MOT17", mode="train"),
#             filter_det(confidence=0.3),
#             normalize(),
#         ],
#         ignore_MOTC=True,
#     )
#     online_train_dset = OnlineTrainingDatasetWrapper(
#         dset=train_dset, skip_first_frame=False
#     )

#     test_dset = MOTChallengeDataset(
#         type="mot-challenge",
#         root="/data/MOT17",
#         mode="test",
#         transforms=[
#             load_app_vectors(
#                 data_root="/data/MOT17", mode="test", feature_extractor="reid"
#             ),
#             load_ecc_transforms(data_root="/data/MOT17", mode="test"),
#             filter_det(confidence=0.3),
#             normalize(),
#         ],
#         ignore_MOTC=True,
#     )
#     online_test_dset = OnlineTrainingDatasetWrapper(
#         dset=test_dset, skip_first_frame=False
#     )

#     return DataLoader(online_train_dset, batch_size=1, shuffle=False), DataLoader(
#         online_test_dset, batch_size=1, shuffle=False
#     )


def main(args):

    model = load_from_ckpt(args.ckpt, args.json_config)
    print(model)

    cfg = ExperimentConfig.parse_file(args.json_config)
    assert isinstance(cfg.dataset, MOTDatasetConfig)
    cfg.dataset.skip_first_frame = False
    train_dloader, val_dloader = cfg.dataset.build_dataloaders()

    dloaders = [val_dloader]
    if args.inference_train:
        dloaders.append(train_dloader)

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=dloaders)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("json_config")
    parser.add_argument("--inference_train", action="store_true")

    args = parser.parse_args()
    main(args)

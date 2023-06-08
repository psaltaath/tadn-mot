from typing import Any, Dict, Tuple

from config4ml.data.dataset import DatasetConfig
from pydantic import validator
from torch.utils.data import Dataset

from ..data.base import OnlineTrainingDatasetWrapper
from ..data.detrac import DetracDataset
from ..data.mot_challenge import MOTChallengeCategorySet, MOTChallengeDataset


# Error handling
class InvalidDatasetError(Exception):
    """Custom error for invalid dataset"""

    pass


class MOTDatasetConfig(DatasetConfig):
    """Base configuration for MOTDataset datasets"""

    ignore_MOTC: bool = False
    load_frame_data: bool = False
    skip_first_frame: bool = True

    def MOTDataset_kwargs(self, mode) -> Dict[str, Any]:
        return {
            "root": self.root,
            "transforms": self.transform_callables(mode=mode),
            "ignore_MOTC": self.ignore_MOTC,
            "load_frame_data": self.load_frame_data,
        }

    @property
    def evaluation_benchmark(self):
        """Returns str identifier of evaluation benchmark"""
        return self.type.upper()


class MOTChallengeDatasetConfig(MOTDatasetConfig):
    """Configuration for MOTChallenge datasets"""

    type = "MOT17"
    detector: str = "FRCNN"
    category_set: MOTChallengeCategorySet = MOTChallengeCategorySet.TRAINING

    @validator("detector")
    def check_detector_type(v):
        if v not in ["FRCNN", "SDP", "DPM", "", "all"]:
            raise InvalidDatasetError(f"Bad detector choice! {v}")
        elif v == "all":
            return ""
        return v

    @validator("category_set", pre=True)
    def check_category_set(v):
        if isinstance(v, MOTChallengeCategorySet):
            return v
        assert isinstance(v, str)

        try:
            v = MOTChallengeCategorySet[v.upper()]
            return v
        except KeyError:
            raise InvalidDatasetError(
                f"{v} is not a valid category set for MOTC datasets"
            )

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train and validation dataset instances

        Raises:
            NotImplementedError: Raises if "split" is invalid

        Returns:
            Dataset: Training dataset instance.
            Dataset: Validation dataset instance.
        """
        if self.split in ["default", "none"]:
            train_dset = MOTChallengeDataset(
                category_set=self.category_set,
                detector=self.detector,
                mode="train",
                version=self.type.upper(),
                **self.MOTDataset_kwargs(mode="train"),
            )

            train_online_dset = OnlineTrainingDatasetWrapper(
                train_dset, skip_first_frame=self.skip_first_frame
            )

            val_dset = MOTChallengeDataset(
                category_set=self.category_set,
                detector=self.detector,
                mode="train",
                version=self.type.upper(),
                **self.MOTDataset_kwargs(mode="val"),
            )

            val_online_dset = OnlineTrainingDatasetWrapper(
                val_dset, skip_first_frame=self.skip_first_frame
            )

            return train_online_dset, val_online_dset
        elif self.split == "half":
            train_dset = MOTChallengeDataset(
                category_set=self.category_set,
                detector=self.detector,
                mode="train_50",
                version=self.type.upper(),
                **self.MOTDataset_kwargs(mode="train"),
            )

            train_online_dset = OnlineTrainingDatasetWrapper(
                train_dset, skip_first_frame=self.skip_first_frame
            )

            val_dset = MOTChallengeDataset(
                category_set=self.category_set,
                detector=self.detector,
                mode="val_50",
                version=self.type.upper(),
                **self.MOTDataset_kwargs(mode="val"),
            )

            val_online_dset = OnlineTrainingDatasetWrapper(
                val_dset, skip_first_frame=self.skip_first_frame
            )

            return train_online_dset, val_online_dset
        else:
            raise NotImplementedError


class DETRACDatasetConfig(MOTDatasetConfig):
    """Configuration for UA-DETRAC datasets"""

    type = "DETRAC"
    detector: str = "EB"

    @validator("detector")
    def check_detector_type(v):
        if v not in ["EB", "frcnn"]:
            raise InvalidDatasetError(f"Bad detector choice! {v}")
        return v

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train and validation dataset instances

        Returns:
            Dataset: Training dataset instance.
            Dataset: Validation dataset instance.
        """

        train_dset = DetracDataset(
            detector=self.detector,
            mode="train",
            **self.MOTDataset_kwargs(mode="train"),
        )

        train_online_dset = OnlineTrainingDatasetWrapper(
            train_dset, skip_first_frame=self.skip_first_frame
        )

        val_dset = DetracDataset(
            detector=self.detector,
            mode="test",
            **self.MOTDataset_kwargs(mode="val"),
        )

        val_online_dset = OnlineTrainingDatasetWrapper(
            val_dset, skip_first_frame=self.skip_first_frame
        )

        return train_online_dset, val_online_dset

    @property
    def evaluation_benchmark(self):
        return "MOT15"


def select_dataset(v: Dict) -> MOTDatasetConfig:
    """Utility function to automatically map dataset to corresponding config"""
    if v["type"].upper() in ["MOT17", "MOT15", "MOT20"]:
        return MOTChallengeDatasetConfig.parse_obj(v)
    elif v["type"].upper() == "DETRAC":
        return DETRACDatasetConfig.parse_obj(v)
    else:
        raise InvalidDatasetError(f"{v['type']} is not a valid dataset type")

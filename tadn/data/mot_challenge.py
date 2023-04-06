from configparser import ConfigParser
from functools import lru_cache
import glob
import os
from enum import Enum, auto
from typing import Callable, Iterable, List

import cv2
import pandas as pd
import numpy as np

from .detection import MOTChallengeDetections

# Local imports
from .base import MOTDataset


class MOTChallengeCategorySet(Enum):
    TARGETS = auto()
    DISTRACTORS = auto()
    OTHER = auto()
    TRAINING = auto()
    ALL = auto()


MOTChallengeCategoryIds = {
    MOTChallengeCategorySet.TARGETS: [1, 2],
    MOTChallengeCategorySet.DISTRACTORS: [7, 8, 12],
    MOTChallengeCategorySet.TRAINING: [1, 2, 7, 8, 12],
    MOTChallengeCategorySet.OTHER: [3, 4, 5, 6, 9, 10, 11],
    MOTChallengeCategorySet.ALL: list(range(1, 13)),
}


class MOTChallengeDataset(MOTDataset):
    """MOT Dataset for UA-Detrac dataset

    Inherits from:
        .base.MOTDataset
    """

    def __init__(
        self,
        *args,
        detector: str = "",
        category_set: MOTChallengeCategorySet = MOTChallengeCategorySet.TRAINING,
        version: str = "MOT17",
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            *args: MOTDataset non keyword args.
            detector (str, optional): Selected object detector. Defaults to "".
            category_set (MOTChallengeCategorySet, optional): Category set to train on. Defaults to MOTChallengeCategorySet.TRAINING.
            version (str, optional): MOTChallenge edition. Defaults to "MOT17".
            **kwargs: MOTDataset keyword args.
        """

        assert detector in ["FRCNN", "SDP", "DPM", ""]
        assert version in ["MOT17", "MOT15"]
        self.version = version
        if self.version != "MOT15":
            self.detector = detector
            self.category_ids = MOTChallengeCategoryIds[category_set]
        else:
            self.detector = ""
            self.category_ids = [-1]

        self.detections_provider = None
        self.has_gt_annotations = True

        super().__init__(*args, **kwargs)

    def _retrieve_sequences(self) -> Iterable[str]:
        """Retrieve available sequences

        Returns:
            Iterable[str]: List of available sequences
        """

        if self.mode in ["train", "train_50", "val_50"]:
            if self.version == "MOT15":
                seqs = glob.glob(os.path.join(self.data_root, "train", "*"))
            else:
                seqs = glob.glob(
                    os.path.join(self.data_root, "train", f"*{self.detector}")
                )
        else:
            seqs = glob.glob(
                os.path.join(self.data_root, self.mode, f"*{self.detector}")
            )
        seqs = filter(os.path.isdir, seqs)

        return seqs

    def _build_db(self):
        """Private method to build dataset"""

        self.detections_provider = MOTChallengeDetections(
            rtv_fun=lambda seq: os.path.join(seq, "det", "det.txt")
        )

        parser = ConfigParser()

        self.db = []

        for seq in self._retrieve_sequences():
            print(f"Building dataset for seq: {seq}")
            parser.read(os.path.join(seq, "seqinfo.ini"))

            frame_height = int(parser.get("Sequence", "imHeight"))
            frame_width = int(parser.get("Sequence", "imWidth"))
            seq_name = parser.get("Sequence", "name")

            seq_length = int(parser.get("Sequence", "seqLength"))

            seq_has_annotations = os.path.exists(os.path.join(seq, "gt", "gt.txt"))
            if seq_has_annotations:
                gt = pd.read_csv(os.path.join(seq, "gt", "gt.txt"), header=None).values  # type: ignore
            else:
                self.has_gt_annotations = False
                gt = None

            if self.mode in ["train", "test"]:
                first_frame = 0
                last_frame = seq_length - 1
            elif self.mode == "train_50":
                first_frame = 0
                last_frame = seq_length // 2 - 1
            elif self.mode == "val_50":
                first_frame = seq_length // 2
                last_frame = seq_length - 1
            else:
                raise AssertionError("Invalid split dataset type")

            for frame_id in range(first_frame, last_frame + 1):
                # det_instance = det[det[:, 0] == frame_id + 1]
                # dets = det_instance[:, 2:7].astype(np.float32)
                dets = self.detections_provider.get(frame_id=frame_id + 1, seq=seq)

                sample_dict = {
                    "frame_height": frame_height,
                    "frame_width": frame_width,
                    "frame_id": frame_id,
                    "seq": seq_name,
                    "is_last_frame_in_seq": False,
                    "seq_first_frame": first_frame,
                }
                sample_dict.update({"detections": dets})

                if seq_has_annotations:
                    assert gt is not None
                    gt_instance = gt[gt[:, 0] == frame_id + 1]
                    # Filter for category type...
                    gt_instance = gt_instance[
                        np.isin(gt_instance[:, 7], self.category_ids)
                    ]
                    track_ids = gt_instance[:, 1].astype(int).tolist()
                    gt_bbs = gt_instance[:, 2:6].astype(np.float32)

                    if len(gt_bbs) == 0:
                        gt_bbs = np.empty((0, 4))

                    motc_gt_file = os.path.join(seq, "gt", "gt.txt")
                    if self.mode == "val_50":
                        motc_gt_file = motc_gt_file.replace("train", "val_50")

                    sample_dict.update(
                        {
                            "gt": gt_bbs,
                            "track_ids": track_ids,
                            "MOTC_gt_file": motc_gt_file,
                        }
                    )
                self.db.append(sample_dict)

            # Set "is_last_frame_in_seq" to True for the last entry of sequence seq
            self.db[-1]["is_last_frame_in_seq"] = True

    def _load_frame_data(self, sample: dict) -> dict:
        """Load frame data

        Args:
            sample (dict): Sample without image data

        Returns:
            dict: Updated sample with image data
        """
        seq = sample["seq"]
        frame_id = sample["frame_id"] + 1

        cfg = ConfigParser()
        cfg.read(os.path.join(self.data_root, self.mode, seq, "seqinfo.ini"))

        imDir = cfg.get("Sequence", "imDir")
        imExt = cfg.get("Sequence", "imExt")

        im_file = os.path.join(
            self.data_root, self.mode, seq, imDir, f"{frame_id:06d}" + imExt
        )
        assert os.path.exists(im_file)
        sample.update(
            {"frame_data": cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)}
        )

        return sample

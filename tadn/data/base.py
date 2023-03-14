from copy import deepcopy
import os
from typing import Iterable, List
import numpy as np
from torch.utils.data import Dataset
import cv2

from .detection import RawDictDetections


class MOTDataset(Dataset):
    """Base class for MOT datasets compatible with TADN.

    Inherits from:
        torch.utils.data.Dataset
    """

    def __init__(
        self,
        root: str,
        transforms: list = [],
        ignore_MOTC: bool = False,
        mode: str = "train",
        load_frame_data: bool = False,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            root (str): Data root folder.
            transforms (list, optional): List of transforms to apply. Defaults to [].
            ignore_MOTC (bool, optional): Don't include path MOTChallenge gt file . Defaults to False.
            mode (str, optional): Dataset mode. Defaults to "train".
            load_frame_data (bool, optional): Flag to read or not image data. Defaults to False.
        """
        super().__init__()

        self.transforms = transforms
        self.data_root = root
        self.ignore_MOTC = ignore_MOTC
        self.mode = mode
        self.load_frame_data = load_frame_data
        self.has_gt_annotations: bool

        self.db: List = []
        self._build_db()

    def _retrieve_sequences(self) -> Iterable[str]:
        """Abstract method to retrieve available sequences

        Raises:
            NotImplementedError: It's an abstract method!
        """
        raise NotImplementedError("Abstract method")

    def _build_db(self) -> None:
        """Abstract method to build dataset

        Raises:
            NotImplementedError: It's an abstract method!
        """
        raise NotImplementedError("Abstract dataset type for MOT")

    def _get_sample(self, index: int) -> dict:
        """Convenience function to retrieve sample by a specified index

        Args:
            index (int): Index of sample

        Returns:
            dict: raw sample
        """
        return self.db[index]

    def __getitem__(self, index: int) -> dict:
        """Retrieve and transform sample for specified index

        Args:
            index (int): Index of sample

        Returns:
            dict: Transformed sample with minimal set of items: {
                'frame_id': (int),
                'gt': np.array,
                'det': np.array,
                'track_ids': list
            }
        """

        sample = deepcopy(self._get_sample(index))
        for T in self.transforms:
            sample = T(sample)

        if self.load_frame_data:
            sample = self._load_frame_data(sample)

        return sample

    def __len__(self) -> int:
        """Returns the total number of samples in dataset.

        Returns:
            int: Total number of samples in dataset
        """
        return len(self.db)

    def _load_frame_data(self, sample: dict) -> dict:
        """Abstract method to load image data

        Args:
            sample (dict): Sample without image data

        Raises:
            NotImplementedError: It's an abstract method!

        Returns:
            dict: Updated sample with image data
        """
        raise NotImplementedError


class OnlineTrainingDatasetWrapper(Dataset):
    """Wrapper class for MOTDataset to use in online training"""

    def __init__(self, dset: MOTDataset, skip_first_frame=True) -> None:
        """Constructor

        Args:
            dset (MOTDataset): Dataset to wrap
            skip_first_frame (bool, optional): Flag whether to return first frame in each sequence. Defaults to True.
        """
        super().__init__()
        self.dset = dset

        def recover_prev(i):
            """Incorporate info regarding past frame"""
            if i == 0:
                return self.dset.db[i]

            prev = self.dset.db[i - 1]["gt"]
            curr = self.dset.db[i]["gt"]
            prev_ids = self.dset.db[i - 1]["track_ids"]
            curr_ids = self.dset.db[i]["track_ids"]

            new_tgt = np.zeros_like(curr_ids, dtype=np.bool)  # type: ignore
            gt_prev = np.zeros_like(curr)

            for idx_curr, t_id_curr in enumerate(curr_ids):
                match_idx = -1
                for idx_prev, t_id_prev in enumerate(prev_ids):
                    if t_id_curr == t_id_prev:
                        match_idx = idx_prev
                if match_idx == -1:
                    new_tgt[idx_curr] = True
                else:
                    gt_prev[idx_curr] = prev[match_idx]

            self.dset.db[i].update({"newborn_tgt": new_tgt, "gt_prev": gt_prev})

            return self.dset.db[i]

        def add_new_sequence_flag(sample):
            """Add flag to sample if it's the first of a sequence"""
            seq_first_frame = (
                sample["seq_first_frame"] + 1
                if skip_first_frame
                else sample["seq_first_frame"]
            )
            sample["new_sequence"] = (
                True if sample["frame_id"] == seq_first_frame else False
            )

        [add_new_sequence_flag(sample) for sample in self.dset.db]

        # Filter-out frame_id == 0 instances
        if self.dset.has_gt_annotations:
            self.dset.db = list(recover_prev(i) for i in range(len(self.dset.db)))
        if skip_first_frame:
            self.dset.db.pop(0)

    def __getitem__(self, index):
        """Retrieve and transform sample for specified index

        Args:
            index (int): Index of sample

        Returns:
            dict: Transformed sample with minimal set of items: {
                'frame_id': (int),
                'gt': np.array,
                'det': np.array,
                'track_ids': list
            }
        """
        return self.dset[index]

    def __len__(self):
        """Returns the total number of samples in dataset.

        Returns:
            int: Total number of samples in dataset
        """
        return len(self.dset)


class SingleVideoDataset(MOTDataset):
    def __init__(
        self,
        video_file: str,
        detections_file: str,
        **kwargs,
    ) -> None:
        assert os.path.exists(video_file)
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)

        # assert os.path.exists(detections_file)
        self.detections_provider = RawDictDetections(detections_file)

        super().__init__(root=video_file, **kwargs)

    def _build_db(self) -> None:
        self.db = []
        # Read video
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        seq_name = os.path.splitext(os.path.basename(self.video_file))[0]

        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for fid in range(frame_count):
            dets = self.detections_provider.get(frame_id=fid)

            sample_dict = {
                "frame_height": frame_height,
                "frame_width": frame_width,
                "frame_id": fid,
                "seq": seq_name,
                "is_last_frame_in_seq": False if fid != (frame_count - 1) else True,
                "seq_first_frame": 0,
            }

            sample_dict.update({"detections": dets})

            self.db.append(sample_dict)

    def _load_frame_data(self, sample: dict) -> dict:
        """Load frame data

        Args:
            sample (dict): Sample without image data

        Returns:
            dict: Updated sample with image data
        """
        frame_id = sample["frame_id"]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        assert ret

        sample.update({"frame_data": frame})

        return sample

    def __del__(self):
        self.cap.release()

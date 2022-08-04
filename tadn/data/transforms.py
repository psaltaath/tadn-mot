import os
import pickle
from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision.ops import nms


class AppVecProvider:
    """Convenience class to load appearance vectors from disk"""

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        feature_extractor: str = "resnet18",
        detector: str = "SDP",
    ) -> None:
        """Constructor

        Args:
            data_root (str): Precomputed appearance features path
            mode (str): Dataset mode [train/val/test/etc...]
            feature_extractor (str, optional): Feature extractor str identifier. Defaults to "resnet18".
            detector (str, optional): Object detector str identifier. Defaults to "SDP".
        """
        self.app_vec_dir = os.path.join(
            data_root, f"appearance_vectors_{feature_extractor}_{detector}_{mode}"
        )

        with open(
            os.path.join(self.app_vec_dir, "ap_vectors.voc"),
            "rb",
        ) as f:
            self.vocabulary = pickle.loads(f.read())
        self.data_root = data_root

    @lru_cache(maxsize=4)
    def _get_app_vec_data(self, app_vec_file: str) -> dict:
        """Private function to read corresponding file from disk and retrieve appearance vector data.

        Args:
            app_vec_file (_type_): _description_

        Returns:
            dict: Dictionary with appearance vector data. Keys correspond to sequence name and frame id.
        """
        vec_data = torch.load(app_vec_file)
        return vec_data

    def __call__(self, sample: dict) -> torch.Tensor:
        """Retrieve appearance features vectors for detections in sample

        Args:
            sample (dict): MOTDataset compatible sample.

        Returns:
            torch.Tensor: (num_detections, app_dim) Set of appearance vectors for sample.
        """
        sample_id = f"{sample['seq']}_{sample['frame_id']}"

        vec_file = os.path.join(self.app_vec_dir, self.vocabulary[sample_id])
        vec_data_dict = self._get_app_vec_data(vec_file)

        return vec_data_dict[sample_id]


class ECCProvider:
    """Convenience class to load ecc transforms from disk"""

    def __init__(self, data_root: str, mode: str = "train") -> None:
        """Constructor

        Args:
            data_root (str): Precomputed ecc path
            mode (str): Dataset mode [train/val/test/etc...]. Defaults to "train".
        """
        self.ecc_dir = os.path.join(data_root, f"ecc_{mode}")

        with open(
            os.path.join(self.ecc_dir, "ecc.pkl"),
            "rb",
        ) as f:
            self.ecc_db = pickle.loads(f.read())
        self.data_root = data_root

    def __call__(self, sample: dict) -> np.ndarray:
        """Retrieve ecc transforms for sample

        Args:
            sample (dict): MOTDataset compatible sample.

        Returns:
            np.ndarray: ECC affine transform matrix
        """
        sample_id = f"{sample['seq']}_{sample['frame_id']}"
        return self.ecc_db[sample_id]


def normalize():
    """Transform to normalize a sample from image coordinates to a unit frame."""

    def apply(sample):
        w = sample["frame_width"]
        h = sample["frame_height"]
        if "gt" in sample:
            sample["gt"] /= np.array([[w, h, w, h]])
        sample["detections"] /= np.array([[w, h, w, h]])

        if "gt_prev" in sample:
            if len(sample["gt_prev"]) == 0:
                sample["gt_prev"] = np.empty((0, 4))
            sample["gt_prev"] /= np.array([[w, h, w, h]])

        if "ecc" in sample:
            sample["ecc"] = _normalize_ecc(sample["ecc"], w, h)
        return sample

    return apply


def _normalize_ecc(M: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convenience function to normalize ecc affine matrix to a unit frame.

    Args:
        M (np.ndarray): Original ecc affine matrix
        w (int): image width
        h (int): image height

    Returns:
        np.ndarray: Normalized ecc affine matrix
    """
    N = np.array([[1 / w, 0, 0], [0, 1 / h, 0]], dtype=np.float32)

    M2 = np.eye(3)

    for A in [cv2.invertAffineTransform(N), M, N]:
        A = np.concatenate([A, np.array([[0, 0, 1]], dtype=A.dtype)], axis=0)
        M2 = A @ M2

    return M2[:2]


def add_noise_det(loc_sigma: float = 15, bb_sigma: float = 5):
    """Transform sample to add Gaussian noise to detection bbox locations.

    Args:
        loc_sigma (float, optional): gaussian sigma for positional components of bbox. Defaults to 15.
        bb_sigma (float, optional): gaussian sigma for size components of bbox. Defaults to 5.
    """

    def apply(sample):
        loc_noise = loc_sigma * np.random.randn(sample["detections"].shape[0], 2)
        bb_noise = bb_sigma * np.random.randn(sample["detections"].shape[0], 2)

        sample["detections"] += np.concatenate([loc_noise, bb_noise], axis=-1)

        return sample

    return apply


def filter_det(confidence: float = 0.3, retain_scores: bool = False):
    """Transform sample to filter detections with confidence lower than threshold.

    Args:
        confidence (float, optional): Confidence threshold. Defaults to 0.3.
        retain_scores (bool, optional): Retain confidence scores in the resulting detections. Defaults to False.
    """
    max_cols = 5 if retain_scores else 4

    def apply(sample):
        if "appearance_vectors" in sample.keys():
            sample["appearance_vectors"] = sample["appearance_vectors"][
                sample["detections"][:, -1] > confidence
            ]

        sample["detections"] = (
            sample["detections"][sample["detections"][:, -1] > confidence]
        )[:, :max_cols]

        return sample

    return apply


def load_app_vectors(
    data_root: str,
    mode: str,
    feature_extractor: str = "resnet18",
    detector: str = "SDP",
):
    """Transform sample to include appearance feature vectors for each detection.

    Args:
        data_root (str): Precomputed appearance features path
        mode (str): Dataset mode [train/val/test/etc...]
        feature_extractor (str, optional): Feature extractor str identifier. Defaults to "resnet18".
        detector (str, optional): Object detector str identifier. Defaults to "SDP".
    """
    app_vec_provider = AppVecProvider(
        data_root, mode=mode, feature_extractor=feature_extractor, detector=detector
    )

    def apply(sample):
        sample.update({"appearance_vectors": app_vec_provider(sample)})
        return sample

    return apply


def load_ecc_transforms(data_root, mode="train"):
    """Transform sample to include ecc affine matrix for each frame.

    Args:
        data_root (str): Precomputed ecc path
        mode (str): Dataset mode [train/val/test/etc...]. Defaults to "train".
    """
    ecc_provider = ECCProvider(data_root, mode)

    def apply(sample):
        sample.update({"ecc": ecc_provider(sample)})
        return sample

    return apply


def nms_det(iou_threshold: float = 0.3, retain_scores=False):
    """Transform sample to filter detections using non-maximum suppression.

    Args:
        iou_threshold (float, optional): IoU threshold. Defaults to 0.3.
        retain_scores (bool, optional): Retain confidence scores in the resulting detections. Defaults to False.
    """
    max_cols = 5 if retain_scores else 4

    def apply(sample):
        assert sample["detections"].shape[-1] == 5

        boxes = np.stack(
            [
                sample["detections"][:, 0],
                sample["detections"][:, 1],
                sample["detections"][:, 0] + sample["detections"][:, 2],
                sample["detections"][:, 1] + sample["detections"][:, 3],
            ],
            axis=1,
        )
        scores = sample["detections"][:, -1]

        selected_boxes = nms(
            torch.from_numpy(boxes),
            torch.from_numpy(scores),
            iou_threshold=iou_threshold,
        ).numpy()

        if "appearance_vectors" in sample.keys():
            sample["appearance_vectors"] = sample["appearance_vectors"][selected_boxes]

        sample["detections"] = (sample["detections"][selected_boxes])[:, :max_cols]

        return sample

    return apply


# Gather all available transforms
def get_all_transforms():
    Ts = [
        normalize,
        add_noise_det,
        filter_det,
        nms_det,
        load_app_vectors,
        load_ecc_transforms,
    ]
    return Ts

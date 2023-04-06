"""Script to precompute appearance features vectors for the whole MOT dataset"""
import glob
import os
import pickle
from argparse import ArgumentParser
from os import path as osp
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchreid
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from ..data.base import OnlineTrainingDatasetWrapper, SingleVideoDataset
from ..data.detrac import DetracDataset

# from .data import OnlineTrainingDataset, load_frame_data
from ..data.mot_challenge import MOTChallengeDataset


class Resnet18Features:
    """Utility class to provide Resnet-18 CNN features"""

    def __init__(self) -> None:
        resnet18 = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(resnet18.children())[:-1], nn.Flatten())
        self.T = Compose(
            [
                ToTensor(),
                Resize((128, 128)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        if len(x) == 0:
            return torch.empty(0)

        batch = torch.stack(list(map(self.T, x)))

        with torch.no_grad():
            out = self.model(batch).cpu()  # BS x 512

        return out


class ReidFeatures:
    """Utility class to provide Resnet-50 ReID CNN features"""

    def __init__(self, ckpt_path: str) -> None:
        """Constructor

        Args:
            ckpt_path (str): Path to pretrained weights.
        """
        self.model = torchreid.utils.FeatureExtractor(
            model_name="resnet50_fc512",
            model_path=ckpt_path,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.preprocess = Compose(
            [
                ToTensor(),
                Resize((256, 128)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x: List[np.ndarray]):
        if len(x) == 0:
            return torch.empty(0)

        batch = torch.stack(list(map(self.preprocess, x)))

        with torch.no_grad():
            out = self.model(batch).cpu()  # BS x 512

        return out


def main(args):
    """Main script function"""
    # base_folder = osp.join(
    # args.data_root,
    # f"appearance_vectors_{args.feature_extractor}_{args.dset_mode}",
    # )
    # print(base_folder)

    torch.set_grad_enabled(False)

    if args.dset_type == "mot-challenge":
        dset = MOTChallengeDataset(
            args.data_root,
            transforms=[],
            ignore_MOTC=True,
            load_frame_data=True,
            mode=args.dset_mode,
            version=args.dset_version,
        )

        base_folder = osp.join(
            args.data_root,
            f"appearance_vectors_{args.feature_extractor}_{args.dset_mode}",
        )
        print(base_folder)

    elif args.dset_type == "detrac":
        dset = DetracDataset(
            args.data_root,
            transforms=[],
            ignore_MOTC=True,
            load_frame_data=True,
            mode=args.dset_mode,
            detector=args.detector,
        )

        base_folder = osp.join(
            args.data_root,
            f"appearance_vectors_{args.feature_extractor}_{args.detector}_{args.dset_mode}",
        )
    elif args.dset_type == "single-video":
        assert os.path.exists(args.data_root)
        assert os.path.exists(args.detections)
        dset = SingleVideoDataset(
            video_file=args.data_root,
            detections_file=args.detections,
            load_frame_data=True,
        )

        base_folder = osp.join(
            os.path.dirname(args.data_root),
            f"{os.path.splitext(os.path.basename(args.data_root))[0]}_appearance_vectors_{args.feature_extractor}",
        )
    else:
        raise Exception("Invalid dataset type")

    if not osp.exists(base_folder):
        os.makedirs(base_folder)

    # dset_wrapper = OnlineTrainingDatasetWrapper(dset, skip_first_frame=False)

    if args.feature_extractor == "resnet18":
        feature_extractor = Resnet18Features()
    else:
        feature_extractor = ReidFeatures(args.reid_ckpt)

    voc_file = osp.join(
        base_folder,
        f"ap_vectors.voc",
    )

    if os.path.exists(voc_file):
        with open(voc_file, "rb") as f:
            feats_vocabulary = pickle.load(f)
        ap_vec_files = glob.glob(
            osp.join(
                base_folder,
                f"ap_vec_*.apv",
            )
        )
        current_file_idx = (
            max(
                list(
                    map(
                        lambda x: int(
                            os.path.splitext(os.path.basename(x))[0].split("_")[-1]
                        ),
                        ap_vec_files,
                    )
                )
            )
            + 1
        )
    else:
        feats_vocabulary = {}
        current_file_idx = 0

    current_file = osp.join(
        base_folder,
        f"ap_vec_{current_file_idx}.apv",
    )

    feats_dict = {}

    key = ""
    try:
        for sample in tqdm(dset):  # type: ignore
            seq = sample["seq"]
            frame_id = sample["frame_id"]

            key = f"{seq}_{frame_id}"

            if key in feats_vocabulary:
                continue

            detections = sample["detections"]

            frame = sample["frame_data"]  # (H, W, 3) uint8 ndarray

            patch_list = []

            # For each detection, crop patch and compute CNN features
            for d in detections:
                real_w: int = int(d[2]) if d[0] > 0 else int(d[0] + d[2])
                real_h: int = int(d[3]) if d[1] > 0 else int(d[1] + d[3])
                real_xmin: int = max(0, int(d[0]))
                real_ymin: int = max(0, int(d[1]))
                if real_h <= 1 or real_w <= 1:
                    patch_list.append(np.zeros((256, 128, 3), dtype=np.uint8))
                else:
                    patch_list.append(
                        frame[
                            real_ymin : real_ymin + real_h,
                            real_xmin : real_xmin + real_w,
                        ]
                    )
            # Inference on a batch!
            feats_tensor = feature_extractor(
                patch_list
            )  # num_dets x 512 @ cuda (if available)

            # Key-value based features storage
            feats_dict.update({key: feats_tensor})
            feats_vocabulary.update({key: osp.basename(current_file)})

            if len(feats_dict.keys()) == args.samples_per_file:
                # Write to file
                torch.save(feats_dict, current_file)
                current_file_idx += 1
                current_file = osp.join(
                    base_folder,
                    f"ap_vec_{current_file_idx}.apv",
                )
                feats_dict = {}
    except Exception as e:
        print("\nProcess interrupted by an exception!")
        print(e)
        print(f"Current key: {key}")
    finally:
        torch.save(feats_dict, current_file)

        voc_file = osp.join(
            base_folder,
            f"ap_vectors.voc",
        )
        with open(voc_file, "wb") as f:
            f.write(pickle.dumps(feats_vocabulary))

    print("Appearance vectors successfully completed!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_root", help="Path to dataset root folder / video file")
    parser.add_argument(
        "--samples_per_file",
        default=1024,
        type=int,
        help="Number of samples to be saved in each storage file",
    )
    parser.add_argument(
        "--dset_type",
        default="mot-challenge",
        type=str,
        choices=["mot-challenge", "detrac", "single-video"],
        help="Dataset type",
    )
    parser.add_argument("--dset_mode", default="train", help="Dataset mode")
    parser.add_argument(
        "--feature_extractor",
        default="resnet18",
        choices=["resnet18", "reid"],
        help="Feature extractor type",
    )
    parser.add_argument(
        "--reid_ckpt",
        default=None,
        help="Path to pretrained reid model. (Only if reid is the feature extractor)",
    )
    parser.add_argument(
        "--dset_version",
        default="MOT17",
        choices=["MOT17", "MOT15"],
        help="Dataset version. Only for MOTChallenge datasets",
    )

    parser.add_argument(
        "--detector",
        default="EB",
        choices=["EB", "frcnn"],
        help="Selected detector. Only for UA-DETRAC.",
    )

    parser.add_argument(
        "--detections",
        default=None,
        help="Path to detections json. Only for single-video mode",
    )

    args = parser.parse_args()

    main(args)

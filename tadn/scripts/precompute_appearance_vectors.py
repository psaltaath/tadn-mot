import pickle
from typing import Tuple
from matplotlib import transforms
import numpy as np
import torchvision.models as models
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
import torch
import torch.nn as nn
from argparse import ArgumentParser
from os import path as osp
import os
from tqdm import tqdm
import torchreid

# from .data import OnlineTrainingDataset, load_frame_data
from ..data.mot_challenge import MOTChallengeDataset
from ..data.detrac import DetracDataset
from ..legacy.carla import CarlaDataset
from ..data.base import OnlineTrainingDatasetWrapper


class Resnet18Features:
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
        x = self.T(x).unsqueeze(0)
        return self.model(x)


class ReidFeatures:
    def __init__(self, ckpt_path) -> None:
        self.model = torchreid.utils.FeatureExtractor(
            model_name="resnet50_fc512",
            model_path=ckpt_path,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def __call__(self, x):
        x = (x * 255).astype(np.uint8)

        return self.model(x)


def main(args):

    torch.set_grad_enabled(False)

    # print(model)

    if args.dset_type == "carla":
        dset = CarlaDataset(
            args.data_root, transforms=[], ignore_MOTC=True, load_frame_data=True
        )
    elif args.dset_type == "mot-challenge":
        dset = MOTChallengeDataset(
            args.data_root,
            transforms=[],
            ignore_MOTC=True,
            load_frame_data=True,
            mode=args.dset_mode,
            version=args.dset_version,
        )
    elif args.dset_type == "detrac":
        dset = DetracDataset(
            args.data_root,
            transforms=[],
            ignore_MOTC=True,
            load_frame_data=True,
            mode=args.dset_mode,
            detector=args.detector,
        )
    else:
        raise Exception("Invalid dataset type")

    dset_wrapper = OnlineTrainingDatasetWrapper(dset, skip_first_frame=False)

    if args.feature_extractor == "resnet18":
        feature_extractor = Resnet18Features()
    else:
        feature_extractor = ReidFeatures(args.reid_ckpt)

    base_folder = osp.join(
        args.data_root,
        f"appearance_vectors_{args.feature_extractor}_{args.detector}_{args.dset_mode}",
    )
    if not osp.exists(base_folder):
        os.makedirs(base_folder)

    current_file_idx = 0
    current_file = osp.join(
        base_folder,
        f"ap_vec_{current_file_idx}.apv",
    )

    feats_vocabulary = {}
    feats_dict = {}

    for sample in tqdm(dset_wrapper):

        seq = sample["seq"]
        frame_id = sample["frame_id"]

        detections = sample["detections"]

        frame = sample["frame_data"].astype(np.float32) / 255

        feat_list = []

        for d in detections:
            real_w: int = int(d[2]) if d[0] > 0 else int(d[0] + d[2])
            real_h: int = int(d[3]) if d[1] > 0 else int(d[1] + d[3])
            real_xmin: int = max(0, int(d[0]))
            real_ymin: int = max(0, int(d[1]))
            if real_h <= 1 or real_w <= 1:
                feat_list.append(torch.zeros(1, 512))
            else:
                feat_list.append(
                    feature_extractor(
                        frame[
                            real_ymin : real_ymin + real_h,
                            real_xmin : real_xmin + real_w,
                        ]
                    )
                )

        if len(feat_list) == 0:
            feats = torch.empty(0)
        else:
            feats = torch.cat(feat_list, dim=0).cpu()

        key = f"{seq}_{frame_id}"
        feats_dict.update({key: feats})
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

    torch.save(feats_dict, current_file)

    voc_file = osp.join(
        base_folder,
        f"ap_vectors.voc",
    )
    with open(voc_file, "wb") as f:
        f.write(pickle.dumps(feats_vocabulary))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--samples_per_file", default=1024, type=int)
    parser.add_argument(
        "--dset_type",
        default="mot-challenge",
        type=str,
        choices=["mot-challenge", "detrac", "carla"],
    )
    parser.add_argument("--dset_mode", default="train")
    parser.add_argument(
        "--feature_extractor", default="resnet18", choices=["resnet18", "reid"]
    )
    parser.add_argument("--reid_ckpt", default=None)
    parser.add_argument("--dset_version", default="MOT17", choices=["MOT17", "MOT15"])

    parser.add_argument("--detector", default="EB", choices=["EB", "frcnn"])

    args = parser.parse_args()

    main(args)

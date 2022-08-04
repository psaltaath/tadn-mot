"""Script to precompute ecc affine transforms for the whole MOT dataset"""
import numpy as np
import cv2
import pickle
import os
from argparse import ArgumentParser
from tqdm import tqdm

from ..data.mot_challenge import MOTChallengeDataset
from ..data.detrac import DetracDataset
from ..data.base import OnlineTrainingDatasetWrapper


def main(args):
    """Main script function"""

    if args.dset_type == "mot-challenge":
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
            detector="EB",
        )
    else:
        raise Exception("Invalid dataset type")

    dset_wrapper = OnlineTrainingDatasetWrapper(dset, skip_first_frame=False)

    transforms = {}

    template_im = None
    target_im = None

    # For each frame, compute ECC affine transform
    for sample in tqdm(dset_wrapper):  # type: ignore

        seq = sample["seq"]
        frame_id = sample["frame_id"]
        key = f"{seq}_{frame_id}"

        if sample["new_sequence"]:
            template_im = cv2.cvtColor(sample["frame_data"], cv2.COLOR_BGR2GRAY)
            transforms[key] = np.eye(2, 3, dtype=np.float32)
            continue

        target_im = cv2.cvtColor(sample["frame_data"], cv2.COLOR_BGR2GRAY)

        _, M = cv2.findTransformECC(template_im, target_im, np.eye(2, 3, dtype=np.float32))  # type: ignore
        transforms[key] = M

        template_im = target_im

    # Save ecc transforms to file "data_root/ecc_[dset_mode]/ecc.pkl"
    current_file = os.path.join(
        args.data_root,
        f"ecc_{args.dset_mode}",
        f"ecc.pkl",
    )
    if not os.path.exists(os.path.join(args.data_root, f"ecc_{args.dset_mode}")):
        os.makedirs(os.path.join(args.data_root, f"ecc_{args.dset_mode}"))

    with open(current_file, "wb") as f:
        pickle.dump(transforms, f)


# Main entry-point
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("data_root", help="Path to dataset root folder")
    parser.add_argument(
        "--dset_type",
        default="mot-challenge",
        type=str,
        choices=["mot-challenge", "detrac"],
        help="Dataset type",
    )
    parser.add_argument("--dset_mode", default="train", help="Dataset mode")
    parser.add_argument(
        "--dset_version",
        default="MOT17",
        choices=["MOT17", "MOT15"],
        help="Dataset version. Only for MOTChallenge datasets",
    )

    args = parser.parse_args()

    main(args)

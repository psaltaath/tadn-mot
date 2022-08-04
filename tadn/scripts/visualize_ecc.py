import cv2
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from ..data.mot_challenge import MOTChallengeDataset
from ..data.base import OnlineTrainingDatasetWrapper
from ..data.transforms import load_ecc_transforms


def main(args):
    dset = MOTChallengeDataset(
        args.data_root,
        transforms=[load_ecc_transforms(args.data_root, mode=args.dset_mode)],
        ignore_MOTC=True,
        load_frame_data=True,
        mode=args.dset_mode,
    )

    dset_wrapper = OnlineTrainingDatasetWrapper(dset, skip_first_frame=False)

    t_w, t_h = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, 20, (t_w, t_h))

    for sample in tqdm(dset_wrapper):
        I = sample["frame_data"]

        w = sample["frame_width"]
        h = sample["frame_height"]

        pt = np.array([w // 2, h // 2])

        M = sample["ecc"]

        pt2 = cv2.transform(pt.reshape(1, 1, 2), M).reshape(-1)

        delta = pt + 15 * (pt2 - pt)

        I = cv2.circle(I, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        I = cv2.line(
            I,
            (int(pt[0]), int(pt[1])),
            (int(delta[0]), int(delta[1])),
            (255, 255, 0),
            2,
        )
        I = cv2.circle(I, (int(delta[0]), int(delta[1])), 15, (255, 0, 0), 3)

        I = cv2.resize(I, (t_w, t_h))
        writer.write(I)
    writer.release()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("output_video", default="ecc_vis.mp4")
    parser.add_argument("--dset_mode", default="train")

    args = parser.parse_args()

    main(args)

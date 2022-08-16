
import pandas as pd
from argparse import ArgumentParser
import cv2
import os
from ..utils.draw import draw_targets


def get_img_detrac(frame_id, seq, data_root, mode):
    im = cv2.imread(
        os.path.join(
            data_root,
            f"Insight-MVT_Annotation_{mode.capitalize()}",
            seq,
            f"img{frame_id+1:05d}.jpg",
        )
    )
    return im


def main(args):
    if args.dset_type == "detrac":
        get_img = lambda fid: get_img_detrac(
            fid, args.seq_name, args.data_root, args.dset_mode
        )
    else:
        raise NotImplementedError

    frame_id = 0
    frame = get_img(frame_id)
    assert frame is not None

    out = cv2.VideoWriter(
        os.path.join(args.results_dir, f"{args.seq_name}.mp4"),
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        25,
        (frame.shape[1], frame.shape[0]),
    )

    results = pd.read_csv(os.path.join(args.results_dir, f"{args.seq_name}.txt")).values

    while frame is not None:
        res_frame = results[results[:, 0] == frame_id + 1]

        ids = res_frame[:, 1]
        bbs = res_frame[:, 2:6]

        frame = draw_targets(ids, bbs, frame)

        out.write(frame)
        frame_id += 1
        frame = get_img(frame_id)
    out.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("seq_name")
    parser.add_argument("results_dir")
    parser.add_argument("data_root")
    parser.add_argument("--dset_type", choices=["detrac"])
    parser.add_argument("--dset_mode", default="test")

    args = parser.parse_args()

    main(args)

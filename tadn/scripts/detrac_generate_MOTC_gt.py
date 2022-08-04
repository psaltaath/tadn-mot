"""Generate MOTChallenge formatted ground truth for UA-DETRAC. For compat with evaluation pipeline"""
import os
from argparse import ArgumentParser

from tqdm import tqdm

from ..data.base import OnlineTrainingDatasetWrapper
from ..data.detrac import DetracDataset


def main(args):
    """Main script function"""
    dset = DetracDataset(
        args.data_root,
        transforms=[],
        ignore_MOTC=True,
        load_frame_data=True,
        mode=args.dset_mode,
        detector="EB",
    )

    motc_out_dir = os.path.join(args.data_root, "motc_gt")
    if not os.path.exists(motc_out_dir):
        os.makedirs(motc_out_dir)

    dset_wrapper = OnlineTrainingDatasetWrapper(dset, skip_first_frame=False)

    motc_template = "{0}, {1}, {2:.1f}, {3:.1f}, {4:.1f}, {5:.1f}, 1, -1, -1, -1\n"
    motc_buffer = []

    for sample in tqdm(dset_wrapper):  # type: ignore

        seq = sample["seq"]
        frame_id = sample["frame_id"]

        for tgt_id, tgt_bbox in zip(sample["track_ids"], sample["gt"]):

            motc_buffer.append(
                motc_template.format(frame_id + 1, tgt_id, *tgt_bbox.tolist())
            )  # frame +1 to switch from 0-index to 1-index

        if sample["is_last_frame_in_seq"]:
            # Write to file
            seq_dir = os.path.join(motc_out_dir, f"{seq}")
            if not os.path.exists(seq_dir):
                os.makedirs(os.path.join(seq_dir, "gt"))
            with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f:
                f.writelines(motc_buffer)
            with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
                f.write(_build_ini(seq, frame_id + 1))
            motc_buffer = []


def _build_ini(seq, seq_len):
    """Utility function to generate .ini files"""

    template = """[Sequence]
name={0}
frameRate=25
seqLength={1}
imWidth=960
imHeight=540
imExt=.jpg"""
    return template.format(seq, seq_len)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_root", help="Path to dataset root folder")
    parser.add_argument("--dset_mode", default="train", help="Dataset mode")
    args = parser.parse_args()

    main(args)

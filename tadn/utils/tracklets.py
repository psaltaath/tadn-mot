from typing import List
import pandas as pd
import numpy as np
from io import StringIO


def truncate_tracklets_MOTC_format(motc_res_lines: List[str]) -> List[str]:
    """Post-process trajectories.
    Truncate after last "hit".
    Delete all trajectories less than 3 frames long

    Args:
        motc_res_lines (List[str]): List of MOTChallenge formatted results lines

    Returns:
        List[str]: Updated list of MOTChallenge formatted results lines
    """
    data = pd.read_csv(StringIO("".join(motc_res_lines)), header=None).values

    ALLOW_TRAJ_DELETION = True
    all_frames = np.unique(data[:, 0])
    if all_frames.max() - all_frames.min() <= 3:
        ALLOW_TRAJ_DELETION = False

    tracklet_ids = np.unique(data[:, 1])

    not_to_be_truncated = np.zeros((data.shape[0]), dtype=bool)

    for trk_id in tracklet_ids:
        tracklet_data = data[data[:, 1] == trk_id]
        if tracklet_data[:, -1].sum() <= 3 and ALLOW_TRAJ_DELETION:
            not_to_be_truncated[data[:, 1] == trk_id] = False
            continue
        tracklets_hits_cum_history = np.flip(np.cumsum(np.flip(tracklet_data[:, -1])))
        not_to_be_truncated[data[:, 1] == trk_id] = tracklets_hits_cum_history != 0

    new_data = data[not_to_be_truncated, :-1]

    str_res = pd.DataFrame(data=new_data).to_csv(header=False, index=False)

    return str_res.splitlines(keepends=True)

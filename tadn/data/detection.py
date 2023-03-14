from copy import copy
from functools import lru_cache
import json
import os
from typing import Callable

import numpy as np
import pandas as pd


class MOTChallengeDetections:
    """Convenience class to manage detection in the MOTChallenge format"""

    def __init__(self, rtv_fun: Callable) -> None:
        """Constructor

        Args:
            rtv_fun (Callable): Function to retrieve detections file path on disk
        """
        self.rtv_fun = rtv_fun

    def get(self, frame_id: int, **kwargs) -> np.ndarray:
        """Retrieve detections for specific frame by frame id.

        Args:
            frame_id (int): Frame id for which detections are requested.
            **kwargs: Misc keyword-arguments to pass on to retrieve function (rtv_fun).

        Returns:
            np.ndarray: (num_dets, 5) detections. Last column is confidence.

        """
        all_dets = self._get_all_dets(**kwargs)

        frame_dets = all_dets[all_dets[:, 0] == frame_id]
        return frame_dets[:, 2:7].astype(np.float32)  # x,y,w,h,conf

    @lru_cache(maxsize=200)
    def _get_all_dets(self, **kwargs) -> np.ndarray:
        """Convenience private function for reading detections

        Args:
            **kwargs: Misc keyword-arguments to pass on to retrieve function (rtv_fun).

        Returns:
            np.ndarray: all detections in raw format
        """
        det_file = self.rtv_fun(**kwargs)

        return pd.read_csv(det_file, header=None).values


class RawDictDetections:
    """
    detections: dict[str, list]

    {
        "frame_id": [
            [
                xmin,
                ymin,
                xmax,
                ymax,
                confidence,
                category
            ],
            [
                ...
            ],
        ]
    }
    """

    def __init__(self, detections_file: str) -> None:
        assert os.path.exists(detections_file)
        with open(detections_file) as f:
            self.dets = json.load(f)

    def get(self, frame_id: int, **kwargs) -> np.ndarray:
        """Retrieve detections for specific frame by frame id.

        Args:
            frame_id (int): Frame id for which detections are requested.
            **kwargs: Misc keyword-arguments to pass on to retrieve function (rtv_fun).

        Returns:
            np.ndarray: (num_dets, 5) detections. Last column is confidence.

        """
        dets_in_frame = self.dets.get(str(frame_id), None)

        if dets_in_frame is None:
            return np.empty((0, 5), dtype=np.float32)
        dets_in_frame = list(map(lambda x: x[:5], dets_in_frame))
        dets_np = np.array(dets_in_frame).astype(np.float32)
        dets_np[:, 2] = dets_np[:, 2] - dets_np[:, 0]  # xmax --> width
        dets_np[:, 3] = dets_np[:, 3] - dets_np[:, 1]  # ymax --> height

        return dets_np

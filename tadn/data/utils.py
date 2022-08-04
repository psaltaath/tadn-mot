from typing import Callable
import pandas as pd
import numpy as np
from functools import lru_cache


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
        return frame_dets[:, 2:7].astype(np.float32)

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

import math

import torch

# Local imports
from . import appearance, motion

# Global values (act as preferences)
_MOTION_MODEL = motion.LinearModel
_APPEARANCE_MODEL = appearance.LastAppearanceVector
_MIN_KILL_THRESHOLD: int = 3
_MAX_KILL_THRESHOLD: int = 30
_MAX_HITS: int = 100


def set_motion_model(model_name: str) -> None:
    """Convenience function to select preferred motion model

    Args:
        model_name (str): ["kalman", "linear"]
    """
    global _MOTION_MODEL
    model_name_dict = {
        motion.LinearModel.type(): motion.LinearModel,
        motion.KalmanModel.type(): motion.KalmanModel,
    }
    assert model_name in model_name_dict
    _MOTION_MODEL = model_name_dict[model_name]


def set_kill_thresholds(min_t: int, max_t: int, max_hits: int = 100):
    """Convenience function to set thresholds

    Args:
        min_t (int): Min kill threshold
        max_t (int): Max kill threshold
        max_hits (int, optional): Max hits for the sigmoid transition function. Defaults to 100.
    """
    global _MIN_KILL_THRESHOLD, _MAX_KILL_THRESHOLD, _MAX_HITS

    _MIN_KILL_THRESHOLD = min_t
    _MAX_KILL_THRESHOLD = max_t
    _MAX_HITS = max_hits


class Tracklet:
    """Tracklet class to represent tracking targets"""

    def __init__(self, id: int, bbox: torch.Tensor, app_vector: torch.Tensor) -> None:
        """Constructor for tracklet

        Args:
            id (int): Target id
            bbox (torch.Tensor): (4, ) Initial bbox location
            app_vector (torch.Tensor): (app_dim, ) Initial appearance features
        """
        self.id = id
        self.motion: motion.AbstractMotionModel = _MOTION_MODEL(bbox)
        self.appearance: appearance.AbstractAppearanceModel = _APPEARANCE_MODEL(
            app_vector
        )

    def update(self, bbox: torch.Tensor, appearance_vector: torch.Tensor) -> None:
        """Updates target state

        Args:
            bbox (torch.Tensor): (4, ) Updated bbox location
            appearance_vector (torch.Tensor): (app_dim, ) Updated appearance features
        """
        self.motion.update(bbox)
        self.appearance.update(appearance_vector)

    @property
    def inactive(self) -> bool:
        """Boolean flag for whether tracklet is in inactive state

        Returns:
            bool: inactive flag
        """

        t_delta = _MAX_KILL_THRESHOLD - _MIN_KILL_THRESHOLD
        norm_hits = self.motion.hits / _MAX_HITS

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        cond_val = round(t_delta * sigmoid(15 * (norm_hits - 0.5)) + 3, 0)

        return self.motion.time_since_update < cond_val

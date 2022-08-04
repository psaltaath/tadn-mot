from typing import Iterable, Tuple
import cv2
import numpy as np
from functools import lru_cache
import random


@lru_cache(maxsize=512)
def random_color(tgt_id: int) -> Tuple[int]:
    """Return a random color.
    Using cache, it returns the same color for up to 512 active targets

    Args:
        tgt_id (int): Target id (used for lru_cache)

    Returns:
        Tuple[int]: RGB triplet
    """
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def draw_targets(ids: Iterable, bbs: Iterable, frame: np.ndarray) -> np.ndarray:
    """Draw targets on a frame

    Args:
        ids (Iterable): Iterable of target ids
        bbs (Iterable): Iterable of bbox locations
        frame (np.ndarray): RGB frame

    Returns:
        np.ndarray: RGB frame with annotations
    """
    for t_id, t_bb in zip(ids, bbs):
        tl = (int(t_bb[0]), int(t_bb[1]))
        br = (int(t_bb[0] + t_bb[2]), int(t_bb[1] + t_bb[3]))
        frame = cv2.rectangle(frame, tl, br, random_color(t_id), 2)
        frame = cv2.putText(
            frame,
            str(t_id),
            (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            random_color(t_id),
            2,
        )
    return frame

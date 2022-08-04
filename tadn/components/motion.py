from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


class AbstractMotionModel(ABC):
    """Abstract base class for Motion models"""

    def __init__(self, bbox: np.ndarray) -> None:
        """Base constructor

        Args:
            bbox (np.ndarray): Initial bbox location
        """
        super().__init__()

        self.hits: int = 0
        self.time_since_update: int = 0
        self.history: list = []
        self.hit_streak: int = 0
        self.age: int = 0
        self.is_hit: bool = True

    @abstractmethod
    def update(self, bbox: np.ndarray) -> None:
        """Update method.
        Updates state using new observation (bbox).

        Args:
            bbox (np.ndarray): Updated bbox location
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.is_hit = True

    @abstractmethod
    def predict(self, ecc_transform: Optional[np.ndarray] = None) -> None:
        """Predict method.
        Updates state using model's predictions. Camera Motion Compensation (CMC) if ECC transform is provided.

        Args:
            ecc_transform (Optional[np.ndarray], optional): ECC affine transform matrix. Defaults to None.
        """
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.is_hit = False

    @property
    @abstractmethod
    def current_state(self) -> np.ndarray:
        """Retrieve current state

        Returns:
            np.ndarray: (4, ) Current bbox location
        """
        pass

    @staticmethod
    @abstractmethod
    def type() -> str:
        """Retrieve str identifier for motion model

        Returns:
            str: motion model type
        """
        pass

    @property
    def hits_ratio(self) -> float:
        """Retrieve hits ratio.
        Percentile of actual detection in tracklet history

        Returns:
            float: hits_ratio
        """
        return self.hits / self.age


class KalmanModel(AbstractMotionModel):
    """Kalman Motion model class"""

    def __init__(self, bbox):
        """Kalman constructor

        Args:
            bbox (np.ndarray): Initial bbox location
        """
        super().__init__(bbox)

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        self.kf.x[:4] = bbox.reshape(-1, 1)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 0.01

    def update(self, bbox):
        """Update method.
        Updates state using new observation (bbox).

        Args:
            bbox (np.ndarray): Updated bbox location
        """
        super().update(bbox)
        self.kf.update(bbox)

    def predict(self, ecc_transform=None):
        """Predict method.
        Updates state using model's predictions. Camera Motion Compensation (CMC) if ECC transform is provided.

        Args:
            ecc_transform (Optional[np.ndarray], optional): ECC affine transform matrix. Defaults to None.
        """
        super().predict()

        if ecc_transform is not None:
            # Invert affine ecc transform to transform to "current" frame coords
            M = cv2.invertAffineTransform(ecc_transform)
            # M = ecc_transform

            # Transfer "old" state vector to new "current" coord frame
            pt = self.kf.x[:2].reshape(1, 1, 2)
            self.kf.x[:2] = cv2.transform(pt, M).reshape(2, 1)

        # if((self.kf.x[6]+self.kf.x[2])<=0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()
        cond = self.kf.x[2:4] < 0.0
        self.kf.x[2:4][cond] = 1e-6
        self.history.append(self.kf.x[:4].squeeze())
        return self.history[-1].astype(np.float32)

    @property
    def current_state(self):
        """Retrieve current state

        Returns:
            np.ndarray: (4, ) Current bbox location
        """
        return self.kf.x[:4].squeeze().astype(np.float32)

    @staticmethod
    def type():
        """Retrieve str identifier for motion model

        Returns:
            str: "kalman"
        """
        return "kalman"


class LinearModel(AbstractMotionModel):
    """Linear Motion model class"""

    def __init__(self, bbox) -> None:
        """Linear MM constructor

        Args:
            bbox (np.ndarray): Initial bbox location
        """
        super().__init__(bbox)
        self.last_bbox = bbox
        self.velocity = np.zeros_like(bbox)

    def update(self, bbox):
        """Update method.
        Updates state using new observation (bbox).

        Args:
            bbox (np.ndarray): Updated bbox location
        """
        self.velocity = (bbox - self.last_bbox) / self.time_since_update
        self.last_bbox = bbox
        super().update(bbox)

    def predict(self, ecc_transform=None):
        """Predict method.
        Updates state using model's predictions. Camera Motion Compensation (CMC) unavailable.

        Args:
            ecc_transform (Optional[np.ndarray], optional): Always set to None. Defaults to None.
        """
        super().predict()

        if ecc_transform is not None:
            raise NotImplementedError("Linear motion model doesn't support ecc")

        prediction = self.last_bbox + self.time_since_update * self.velocity
        cond = prediction[2:] < 0.0
        prediction[2:][cond] = 1e-6
        self.history.append(prediction)
        return self.history[-1]

    @property
    def current_state(self):
        """Retrieve current state

        Returns:
            np.ndarray: (4, ) Current bbox location
        """
        if self.time_since_update == 0:
            return self.last_bbox
        else:
            return self.history[-1]

    @staticmethod
    def type():
        """Retrieve str identifier for motion model

        Returns:
            str: "linear"
        """
        return "linear"

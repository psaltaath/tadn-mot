from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..components import Tracklet
from ..components.transformer import TADN

# List of available colours
COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
]


class AbstractManager(nn.Module):
    """Base class for MOT managers"""

    def __init__(self, tracker: TADN, app_dim: int = 512) -> None:
        """Constructor

        Args:
            tracker (TADN): TADN compatible tracking module
            app_dim (int, optional): Appearance features dimensionality. Defaults to 512.
        """

        super().__init__()

        self.tracker = tracker

        self.app_dim = app_dim

        self.init_hyperparameters: dict[str, Any] = {
            "app_dim": app_dim,
        }

        self.tracklets: List[Tracklet] = list()

        self.lam_validation: bool = False

    @property
    def device(self):
        """Return manager's current device"""
        return next(self.tracker.parameters()).device

    def step(self, detections: torch.Tensor):
        """Abstract method for manager step

        Args:
            detections (torch.Tensor): (num_detections, 4) Detections

        Raises:
            NotImplementedError: Abstract method
        """
        raise NotImplementedError("Abstract class")

    def draw_targets(self, frame: np.ndarray) -> np.ndarray:
        """Convenience method to draw active targets on a frame

        Args:
            frame (np.ndarray): RGB Image

        Returns:
            np.ndarray: RGB Image with active targets drawn
        """
        with torch.no_grad():
            for tgt in self.tracklets:
                tgt_bb = tgt.motion.current_state
                tgt_real_id = tgt.id
                tl = (int(tgt_bb[0] * frame.shape[1]), int(tgt_bb[1] * frame.shape[0]))
                br = (
                    int((tgt_bb[0] + tgt_bb[2]) * frame.shape[1]),
                    int((tgt_bb[1] + tgt_bb[3]) * frame.shape[0]),
                )
                cv2.rectangle(frame, tl, br, COLORS[tgt_real_id % len(COLORS)], 2)
                cv2.putText(
                    frame,
                    str(tgt_real_id),
                    (tl[0], tl[1] - 10),
                    0,
                    0.5,
                    COLORS[tgt_real_id % len(COLORS)],
                    2,
                )

        return frame

    def similarity_matrix_hook(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Register a new similarity matrix

        Args:
            similarity_matrix (torch.Tensor): Similarity matrix to be registered

        Returns:
            torch.Tensor: Similarity matrix
        """
        return similarity_matrix

    @property
    def current_state(self) -> dict:
        """Returns the current state of tracked trajectories.
        For compatibility with clear-MOT metrics

        Returns:
            dict: Manager's state as dict with items:
                {tracklet_id:{
                    "bbox":bbox location,
                    "is_hit":flag if last position corresponds to detection
                    }
                }
        """
        state = {
            trk.id: {
                "bbox": torch.tensor(
                    trk.motion.current_state, device=self.device, requires_grad=False
                ),
                "is_hit": trk.motion.is_hit,
            }
            for trk in self.tracklets
        }

        return state

    @property
    def track_locations(self) -> torch.Tensor:
        """Get bbox locations for all active targets

        Returns:
            torch.Tensor: (num_targets, 4) bbox locations
        """
        if len(self.tracklets) == 0:
            return torch.empty(0, 4, device=self.device, requires_grad=False)

        track_locations = torch.stack(
            [
                torch.from_numpy(trk.motion.current_state).to(self.device)
                for trk in self.tracklets
            ]
        )
        track_locations.requires_grad = False

        return track_locations

    def motion_predictions(
        self, ecc_transform: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Predict future bbox locations for active targets using motion model.

        Args:
            ecc_transform (np.ndarray, optional): ECC affine transform matrix for CMC. Defaults to None.

        Returns:
            torch.Tensor: (num_targets, 4) Predicted bbox locations
        """
        if len(self.tracklets) == 0:
            return torch.empty(0, 4, device=self.device, requires_grad=False)

        tgt_motion_model_preds = torch.stack(
            [
                torch.from_numpy(trk.motion.predict(ecc_transform=ecc_transform)).to(
                    self.device
                )
                for trk in self.tracklets
            ]
        )
        tgt_motion_model_preds.requires_grad = False
        return tgt_motion_model_preds

    @property
    def appearance_vectors(self) -> torch.Tensor:
        """Get appearance features vectors for all active targets

        Returns:
            torch.Tensor: (num_targets, app_dim) appearance features vectors
        """
        if len(self.tracklets) == 0:
            return torch.empty(0, self.app_dim, device=self.device)

        app_vecs = torch.stack(
            [
                torch.from_numpy(trk.appearance.current_state).to(self.device)
                for trk in self.tracklets
            ]
        )
        return app_vecs


class GenericManager(AbstractManager):
    """Generic MOT manager class

    Inherits from:
        AbstractManager
    """

    def _update_tracks(
        self,
        assigned_dets: Optional[torch.Tensor],
        assigned_tgts: Optional[torch.Tensor],
        detections: torch.Tensor,
        det_app_vectors: torch.Tensor,
    ) -> None:
        """Private method to update targets

        Args:
            assigned_dets (torch.Tensor, optional): (K, ) Assigned detections indices
            assigned_tgts (torch.Tensor, optional): (K, ) Assigned targets indices
            detections (torch.Tensor): (num_detections, 4) Detections
            det_app_vectors (torch.Tensor): (num_detections, app_dim) Appearance features vectors for detections
        """

        if assigned_tgts is None:
            return
        assert assigned_dets is not None

        assigned_tracklets = [self.tracklets[i] for i in assigned_tgts]
        for trk, det_idx in zip(assigned_tracklets, assigned_dets):
            trk.update(
                bbox=detections[det_idx].cpu().numpy(),
                appearance_vector=det_app_vectors[det_idx].cpu().numpy(),
            )

    def _perform_assignments(
        self, similarity_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute assignments using the Hungarian algorithm

        Args:
            similarity_matrix (torch.Tensor): (num_detections, num_targets) Similarity matrix

        Returns:
            torch.Tensor: assigned detection indices
            torch.Tensor: assigned target indices
        """
        det_ids, tgt_ids = linear_sum_assignment(
            similarity_matrix.cpu().numpy(), maximize=True
        )
        return torch.from_numpy(det_ids).type(torch.long), torch.from_numpy(
            tgt_ids
        ).type(torch.long)

    def _pre_step(
        self,
        detections: torch.Tensor,
        det_app_vectors: torch.Tensor,
        computed_motion_predictions: torch.Tensor,
    ) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Handles assignments computation, but does NOT perform any changes to track_history etc.

        Args:
            detections (torch.Tensor): (num_detections, 4) Detections
            det_app_vectors (torch.Tensor): (num_detections, app_dim) Appearance features vectors for detections
            computed_motion_predictions (torch.Tensor): (num_targets, 4). Predicted target locations
        Returns:
            np.ndarray[bool]: (num_detections, ) Which detections have been used during pre_step
            torch.Tensor, optional: assigned detection indices
            torch.Tensor, optional: assigned target indices
        """

        assert detections.size()[1] == 4
        used_detections = np.zeros(detections.size()[0]).astype(bool)

        # Check if targets exist
        if len(self.tracklets) == 0 or len(detections) == 0:
            return used_detections, None, None

        assert computed_motion_predictions is not None

        # Compute similarity across targets and detections
        similarity_matrix = self.tracker.forward(
            targets=computed_motion_predictions,
            track_app_vec=self.appearance_vectors,
            detections=detections,
            det_app_vectors=det_app_vectors,
        )

        # Access future-proof hook (for subclassing)
        if self.training or self.lam_validation:
            similarity_matrix = self.similarity_matrix_hook(similarity_matrix)

        if similarity_matrix.numel() == 0:
            return used_detections, None, None

        # Find which detections are directly assigned by the tracker to the NULL target.
        # Use those to "spawn" NEW TARGETS via the "used_detections" array
        dets_not_assigned_to_null = (
            similarity_matrix.argmax(dim=-1) != similarity_matrix.size()[-1] - 1
        )
        dets_lut = torch.arange(len(dets_not_assigned_to_null))[
            dets_not_assigned_to_null
        ]

        used_detections[dets_not_assigned_to_null.cpu().numpy()] = True

        # Remove null-assigned dets from sm.
        # These "unused" detections will be used to generate new targets
        similarity_matrix = similarity_matrix[dets_not_assigned_to_null]

        # Calculate linear_sum_assignment
        assigned_valid_dets, assigned_valid_tgts = self._perform_assignments(
            similarity_matrix
        )

        # Convert back to "dense" indices
        assigned_dets = dets_lut[assigned_valid_dets]
        assigned_tgts = torch.from_numpy(assigned_valid_tgts)

        return used_detections, assigned_dets, assigned_tgts

    def _exec_step(
        self,
        detections: torch.Tensor,
        used_detections: np.ndarray,
        assigned_dets: Optional[torch.Tensor],
        assigned_tgts: Optional[torch.Tensor],
        det_app_vectors: torch.Tensor,
    ) -> None:
        """Perform assignments and update manager state

        Args:
            detections (torch.Tensor): (num_detections, 4) Detections
            used_detections (np.ndarray): Which detections have been used during pre_step
            assigned_dets (torch.Tensor): assigned detection indices
            assigned_tgts (torch.Tensor): assigned target indices
            det_app_vectors (torch.Tensor): (num_detections, app_dim) Appearance features vectors for detections
        """

        if len(self.tracklets) > 0:
            # Update tracks
            self._update_tracks(
                assigned_dets, assigned_tgts, detections, det_app_vectors
            )

        # Kill targets
        self.tracklets = list(
            filter(
                lambda trk: trk.inactive,
                self.tracklets,
            )
        )

        # Birth of new targets
        if np.logical_not(used_detections).sum() > 0:
            # Create a new target for-each unused detection
            for unused_detection, unused_app_vec in zip(
                detections[np.logical_not(used_detections)],
                det_app_vectors[np.logical_not(used_detections)],
            ):
                new_idx = 0
                if len(self.tracklets) > 0:
                    new_idx = max((trk.id for trk in self.tracklets)) + 1

                new_tracklet = Tracklet(
                    id=new_idx,
                    bbox=unused_detection.clone().detach().cpu().numpy(),
                    app_vector=unused_app_vec.clone().detach().cpu().numpy(),
                )
                # Birth!!!!
                self.tracklets.append(new_tracklet)

    def step(
        self,
        detections: torch.Tensor,
        det_app_vectors: torch.Tensor,
        computed_motion_predictions: torch.Tensor,
    ) -> None:
        """Perform a full manager step

        Args:
            detections (torch.Tensor): (num_detections, 4) Detections
            det_app_vectors (torch.Tensor): (num_detections, app_dim) Appearance features vectors for detections
            computed_motion_predictions (torch.Tensor): (num_targets, 4). Predicted target locations
        """

        assert computed_motion_predictions is not None

        used_detections, assigned_dets, assigned_tgts = self._pre_step(
            detections=detections,
            det_app_vectors=det_app_vectors,
            computed_motion_predictions=computed_motion_predictions,
        )

        self._exec_step(
            detections=detections,
            used_detections=used_detections,
            assigned_dets=assigned_dets,
            assigned_tgts=assigned_tgts,
            det_app_vectors=det_app_vectors,
        )


class ModelAssignmentManager(GenericManager):
    """MOT Manager with support for TADN assignment method"""

    def _perform_assignments(
        self, similarity_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            similarity_matrix (torch.Tensor): (num_detections, num_targets) Similarity matrix

        Returns:
            torch.Tensor: assigned detection indices
            torch.Tensor: assigned target indices
        """

        if similarity_matrix.numel() == 0:
            return torch.empty(
                0, device=similarity_matrix.device, dtype=torch.long
            ), torch.empty(0, device=similarity_matrix.device, dtype=torch.long)
        # Calculate detection assignments given max score (Det2Tgt)
        a_det2tgt = similarity_matrix.argmax(dim=-1)

        # Serialize indices to refer to flattened similarity matrix
        sm_serializer = torch.linspace(
            0,
            similarity_matrix.numel() - similarity_matrix.size()[1],
            steps=similarity_matrix.size()[0],
            device=a_det2tgt.device,
            dtype=torch.int,
        )
        a_det2tgt_fl = a_det2tgt + sm_serializer

        # Compute assignment costs
        a_costs = similarity_matrix.view(-1)[a_det2tgt_fl]

        # Compute (putative) assignment pairs [D,T,S]
        # a [det_id, tgt_id, ass_score] {nx3} --T--> {3xn}
        a = torch.stack(
            [
                torch.arange(similarity_matrix.size()[0], device=a_det2tgt.device),
                a_det2tgt,
                a_costs,
            ],
            dim=-1,
        ).transpose(0, 1)

        def select_best_detection_for_assignment(tgt_idx):
            """Filter duplicate assignments to same target using best assignment score"""
            part_a = a.transpose(0, 1)[a[1] == tgt_idx].transpose(0, 1)
            max_idx = part_a[-1].argmax()
            return part_a.transpose(0, 1)[max_idx]

        # Unassign multiple detections to single target
        a = torch.stack(
            list(
                [
                    select_best_detection_for_assignment(tgt_idx)
                    for tgt_idx in a[1].unique()
                ]
            )
        ).transpose(0, 1)

        return a[0].type(torch.long), a[1].type(torch.long)

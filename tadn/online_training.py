import os
import random
from typing import Any, Dict, Optional, Tuple
import hydra
from omegaconf import DictConfig
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

# Local imports
from .utils.convenience import get_trainer
from .utils.convenience import get_evaluation_benchmark, get_tracker
from .components import tracklets
from .components.transformer import TADN
from .config.data import MOTDatasetConfig
from .config.experiment import ExperimentConfig
from .mot import metrics
from .mot.eval import MOTEvaluator, MOTInference
from .mot.managers import ModelAssignmentManager
from .utils.bbox import bbox_xywh2xyxy, convert_MOTC_format
from .utils.scheduler import SigmoidScheduler
from .utils.tracklets import truncate_tracklets_MOTC_format


class OnlineManager(ModelAssignmentManager):
    """Adaptation of TADN MOT Manager for online training

    Inherits from:
        ModelAssignmentManager
    """

    def __init__(
        self,
        *args,
        choice_assignment_params: dict = {"starting_epoch": 5, "ending_epoch": 40},
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            choice_assignment_params (dict, optional): Choice assignment options. Defaults to {"starting_epoch": 5, "ending_epoch": 40}.
            *args: ModelAssignmentManager standard positional args
            **kwargs: ModelAssignmentManager standard keyword args
        """

        super().__init__(*args, **kwargs)

        self.current_epoch = -1
        self.assignment_loss = torch.tensor(
            0.0, dtype=torch.float32, requires_grad=True
        )

        self.gt_bias_probability = SigmoidScheduler(
            starting_value=1,
            target_value=0,
            starting_epoch=choice_assignment_params["starting_epoch"],
            ending_epoch=choice_assignment_params["ending_epoch"],
        )

        self.init_hyperparameters.update(
            {"choice_assignment_params": choice_assignment_params}
        )

        self.null_target_weight = 1.0

    def reset(self):
        """Reset manager state.
        Clear tracklet history, clean up
        """
        self.tracklets = list()

        self.label_assignment_matrix = None
        self.zero_loss()

    def zero_loss(self) -> None:
        """Set assignment loss to zero"""
        self.assignment_loss = torch.tensor(
            0.0, dtype=torch.float32, requires_grad=True
        )

    def similarity_matrix_hook(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Register and update similarity matrix
        Choose between given similarity matrix and label assignment matrix
        to output a new similarity matrix

        Args:
            similarity_matrix (torch.Tensor): (num_detections, num_targets+1) Similarity matrix between detections and targets

        Returns:
            torch.Tensor: (num_detections, num_targets+1) Updated Similarity matrix
        """
        assert self.label_assignment_matrix is not None
        c_weights = torch.ones(similarity_matrix.size()[-1], device=self.device)
        c_weights[-1] = self.null_target_weight
        c_weights /= torch.norm(c_weights)

        if self.label_assignment_matrix.numel() == 0:
            self.assignment_loss = torch.tensor(
                0.0, device=self.device, requires_grad=True
            )
            choice_sm = self.label_assignment_matrix.clone()
            self.label_assignment_matrix = None
            return choice_sm

        self.assignment_loss = F.nll_loss(
            input=torch.log_softmax(similarity_matrix, dim=-1),
            target=self.label_assignment_matrix.argmax(dim=-1),
            reduction="mean",
            weight=c_weights,
        )
        self.assignment_loss /= len(self.tracklets)

        # Select between gt and model
        choice = torch.rand(
            (similarity_matrix.size()[0]), device=similarity_matrix.device
        ) < self.gt_bias_probability.step(self.current_epoch)

        choice_sm = torch.where(
            choice.view(-1, 1), self.label_assignment_matrix, similarity_matrix
        )

        self.label_assignment_matrix = None
        return choice_sm

    def register_label_assignment_matrix(
        self, label_assignment_matrix: torch.Tensor, epoch: int
    ) -> None:
        """Register LAM from training

        Args:
            label_assignment_matrix (torch.Tensor): (num_detections, num_targets+1) Label Assignment Matrix (LAM)
            epoch (int): current epoch of training for choice probability
        """
        self.label_assignment_matrix = label_assignment_matrix.type(torch.float32)
        self.current_epoch = epoch


class OnlineTraining(pl.LightningModule):
    """Module to train TADN using the online training strategy"""

    def __init__(
        self,
        manager: OnlineManager,
        tgt2det_min_threshold: float = 0.3,
        null_target_weight: int = 1,
        learning_rate: float = 0.0001,
        allow_reflection: bool = True,
        lr_scheduler_params: Dict[str, Any] = {
            "type": "StepLR",
            "step_size": 80,
            "gamma": 0.1,
        },
        assignment_threshold: float = 0.1,
        benchmark="MOT17",
    ):
        """Constructor

        Args:
            manager (OnlineManager): OnlineManager instance
            tgt2det_min_threshold (float, optional): Target to detections minimum threshold. Defaults to 0.3.
            null_target_weight (int, optional): Weight for the null-target. Defaults to 1.
            learning_rate (float, optional): Learning rate. Defaults to 0.0001.
            allow_reflection (bool, optional): Flag for reflection augmentation. Defaults to True.
            lr_scheduler_params (_type_, optional): Learning rate scheduler config. Defaults to { "type": "StepLR", "step_size": 80, "gamma": 0.1, }.
            assignment_threshold (float, optional): Assignment threshold. Defaults to 0.1.
            benchmark (str, optional): Benchmark. Defaults to "MOT17".
        """

        super().__init__()
        self.learning_rate = learning_rate

        self.manager: OnlineManager = manager
        self.manager.null_target_weight = null_target_weight

        self.tgt2det_min_threshold = tgt2det_min_threshold

        self.null_target_weight = null_target_weight

        # Temp variables
        self.used_detections = None
        self.assigned_dets = None
        self.assigned_tgts = None
        self.detections = None
        self.tgt_motion_model_preds = None
        self.allow_reflection = allow_reflection

        self.init_hyperparameters = {
            "tgt2det_min_threshold": tgt2det_min_threshold,
            "null_target_weight": null_target_weight,
            "allow_reflection": allow_reflection,
            "lr_scheduler_params": lr_scheduler_params,
        }

        # training opts
        self.reflect_x: Optional[bool] = None
        self.reflect_y: Optional[bool] = None

        self.lr_scheduler_params = lr_scheduler_params

        self.assign_threshold = assignment_threshold

        self.val_res_buffer = list()
        self.results_file = None
        self.evaluator = MOTEvaluator(benchmark=benchmark)
        self.inferencer = MOTInference(benchmark=benchmark)

    @property
    def device(self) -> torch.device:
        """Get current device"""
        return next(self.parameters()).device

    def _new_sequence(self):
        """Method to call when a new sequence is fed."""
        self.manager.reset()

        if self.allow_reflection:
            self.reflect_x = random.uniform(0, 1) > 0.5
            self.reflect_y = random.uniform(0, 1) > 0.5
        else:
            self.reflect_x = False
            self.reflect_y = False

    def _reflect_sample(self, batch):
        """
        Reflect input samples according to self-variables "reflect_x/y"
        """
        if self.reflect_x:
            batch["detections"][:, :, 0] = (1 - batch["detections"][:, :, 0]) - batch[
                "detections"
            ][:, :, 2]
            batch["gt"][:, :, 0] = (1 - batch["gt"][:, :, 0]) - batch["gt"][:, :, 2]
            batch["gt_prev"][:, :, 0] = (1 - batch["gt_prev"][:, :, 0]) - batch[
                "gt_prev"
            ][:, :, 2]
        if self.reflect_y:
            batch["detections"][:, :, 1] = (1 - batch["detections"][:, :, 1]) - batch[
                "detections"
            ][:, :, 3]
            batch["gt"][:, :, 1] = (1 - batch["gt"][:, :, 1]) - batch["gt"][:, :, 3]
            batch["gt_prev"][:, :, 1] = (1 - batch["gt_prev"][:, :, 1]) - batch[
                "gt_prev"
            ][:, :, 3]

        return batch

    def _assign_gt_to_motm(
        self,
        gt_tgt_locs: torch.Tensor,
        gt_tgt_locs_prev: torch.Tensor,
        detections: torch.Tensor,
        motm_target_loc_prev: torch.Tensor,
        motm_target_loc_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Label Assignment Matrix (LAM)

        Args:
            gt_tgt_locs (torch.Tensor): (num_gt_targets, 4) Current gt targets bbx locations
            gt_tgt_locs_prev (torch.Tensor): (num_gt_targets, 4) Previous gt targets bbx locations
            detections (torch.Tensor): (num_detections, 4) Detections in current frame
            motm_target_loc_prev (torch.Tensor): (num_motm_targets, 4) Previous motm targets bbx locations
            motm_target_loc_predictions (torch.Tensor): (num_motm_targets, 4) Current motm targets bbx locations
        Outputs:
            torch.Tensor: ID associations between GT and MOTM
            torch.Tensor: (num_detections, num_motm_targets+1) LAM
        """
        assert gt_tgt_locs.size()[0] == gt_tgt_locs_prev.size()[0]

        if motm_target_loc_prev.size()[0] == 0:
            return torch.empty((0, 3), device=self.device), torch.empty(
                (0, 0), device=self.device
            )

        if gt_tgt_locs_prev.size()[0] == 0:
            return torch.empty((0, 3), device=self.device), torch.empty(
                (0, 0), device=self.device
            )

        # Use previous to compute id matches between GT and MOTM
        gt_motm_similarity_matrix: np.ndarray = (
            metrics.pairwise(
                bbox_xywh2xyxy(gt_tgt_locs_prev), bbox_xywh2xyxy(motm_target_loc_prev)
            )
            .cpu()
            .numpy()
        )

        gt_ind, motm_ind = linear_sum_assignment(
            gt_motm_similarity_matrix, maximize=True
        )

        # Note "gt_motm_similarity_matrix" has columns compatible to "active targets" order!

        # Compute MAX iou for each MOTM tgt relative to GT tgts
        # putative_assignments: (#MOTM active tgts, 3)
        #   Col#1-->Indexing relative to MOTM active targets,
        #   Col#2-->GT tgt with max iou to each MOTM tgt,
        #   Col#3-->Assignment cost/iou value

        putative_assignments_list: list = []
        for gt_idx, motm_idx in zip(gt_ind, motm_ind):
            putative_assignments_list.append(
                [
                    motm_idx,
                    gt_idx,
                    float(gt_motm_similarity_matrix[gt_idx, motm_idx]),
                ]
            )
        putative_assignments = torch.tensor(
            putative_assignments_list, dtype=torch.float32, device=self.device
        )

        # Filter out assignments with cost less than threshold
        on_track_assignments = putative_assignments[
            putative_assignments[:, 2] > self.assign_threshold
        ]

        unassigned_gt = list(
            set(list(range(gt_tgt_locs.size()[0])))
            - set(on_track_assignments[:, 1].cpu().tolist())
        )
        unassigned_motm = list(
            set(list(range(len(motm_target_loc_prev))))
            - set(on_track_assignments[:, 0].cpu().tolist())
        )

        # Compute the ID assignment matrix.
        # Use "-1" for the NO ASSIGNMENT (missing/inactive) index
        id_assignments = list()
        for a in on_track_assignments:
            id_assignments.append((int(a[1]), int(a[0])))
        for gt_idx in unassigned_gt:
            id_assignments.append((int(gt_idx), -1))
        for motm_idx in unassigned_motm:
            id_assignments.append((-1, int(motm_idx)))
        id_assignments = torch.tensor(id_assignments).to(self.device)

        # Compute Label Assignment Matrix (LAM)

        on_track_and_inactive_locs = bbox_xywh2xyxy(
            torch.cat(
                [
                    gt_tgt_locs[on_track_assignments[:, 1].type(torch.long)],
                    motm_target_loc_predictions[unassigned_motm],
                ],
                dim=0,
            )
        )

        # Sort to "active target" indexing

        indices_unsorted = torch.cat(
            (on_track_assignments[:, 0], torch.tensor(unassigned_motm).to(self.device))
        ).type(torch.long)

        # Assert that collected indices equal to active targets
        assert (
            torch.sort(indices_unsorted).values
            == torch.arange(motm_target_loc_prev.size()[0], device=self.device)
        ).all()

        active_tgt_predicted_locs = on_track_and_inactive_locs[
            torch.argsort(indices_unsorted)
        ]

        # Compute pairwise IoU metric
        LAM = metrics.pairwise(bbox_xywh2xyxy(detections), active_tgt_predicted_locs)
        criterion1 = LAM == LAM.max(dim=-1).values.unsqueeze(-1).expand_as(LAM)
        criterion2 = LAM >= self.tgt2det_min_threshold
        LAM = criterion1 & criterion2

        # Add NULL target for detections with IoU < THRESH for every target
        nt_column = torch.logical_not(LAM).all(dim=-1).view((LAM.size()[0], 1))
        LAM = torch.cat((LAM, nt_column), dim=-1)

        return id_assignments, LAM

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a training step
        batch_data  --> Detections Tensor nx4 [xmin,ymin,xmax,ymax] normalized
                    --> GT_Targets Tensor nx4 [xmin,ymin,xmax,ymax] normalized
                    --> GT_Targets_last_frame Tensor nx4 normalized
                    --> GT_assignments Tuple (list_of_det_idx, list_of_tgt_idx)
                    --> GT_orphan_dets list_of_det_idx
                    --> New Sequence bool [Reset MOT?]
                    [-->] ECC transforms (Optional)

        Args:
            batch (torch.Tensor): input batch of data
            batch_idx (int): batch index

        Returns:
            torch.Tensor: loss to be optimized
        """
        self.manager.zero_loss()
        assert batch["detections"].size()[0] == 1  # Batch-size must be 1!

        # Reflect batch!
        if batch["new_sequence"]:
            self._new_sequence()

        batch = self._reflect_sample(batch)

        detections = batch["detections"].flatten(0, 1)
        gt_tgt_locs = batch["gt"].flatten(0, 1)
        gt_tgt_locs_prev = batch["gt_prev"].flatten(0, 1)

        det_app_vectors = batch["appearance_vectors"].flatten(0, 1)

        ecc_transform = batch["ecc"][0].cpu().numpy() if "ecc" in batch else None

        # Precompute Motion Model predictions
        tgt_motion_model_preds = self.manager.motion_predictions(
            ecc_transform=ecc_transform
        )
        motm_target_loc_prev = self.manager.track_locations

        # Assign gt_tgts to MOT tgts via indices
        id_assignments, label_am = self._assign_gt_to_motm(
            gt_tgt_locs,
            gt_tgt_locs_prev,
            detections,
            motm_target_loc_prev,
            tgt_motion_model_preds,
        )

        # Register to MOT Manager the "LABEL" assignment matrix (DxN+1, N:active targets)
        self.manager.register_label_assignment_matrix(
            label_am, epoch=self.current_epoch
        )

        used_detections, assigned_dets, assigned_tgts = self.manager._pre_step(
            detections=detections,
            det_app_vectors=det_app_vectors,
            computed_motion_predictions=tgt_motion_model_preds,
        )

        self.used_detections = used_detections
        self.assigned_dets = assigned_dets
        self.assigned_tgts = assigned_tgts

        total_loss = self.manager.assignment_loss

        # Log metrics
        self.log(
            "assignment_cross_entropy_loss",
            self.manager.assignment_loss.detach().cpu(),
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "composite_loss", total_loss.detach().cpu(), on_epoch=True, batch_size=1
        )
        return total_loss

    def on_train_batch_end(self, outputs: Any, batch: dict, batch_idx: int) -> None:
        """Actions to be performed after each training step

        Args:
            outputs (Any): Training outputs
            batch (torch.Tensor): input batch of data
            batch_idx (int): batch index
        """
        det_app_vectors = batch["appearance_vectors"].flatten(0, 1)
        detections = batch["detections"].flatten(0, 1)

        assert self.used_detections is not None
        self.manager._exec_step(
            detections=detections,
            used_detections=self.used_detections,
            assigned_dets=self.assigned_dets,
            assigned_tgts=self.assigned_tgts,
            det_app_vectors=det_app_vectors,
        )

        self.used_detections = None
        self.assigned_dets = None
        self.assigned_tgts = None
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        """Actions to be performed after each training epoch ends"""
        self.log(
            "choice_assign",
            float(self.manager.gt_bias_probability.step(self.current_epoch)),
            batch_size=1,
            on_epoch=True,
            on_step=False,
        )
        return super().on_train_epoch_end()

    def _test_or_val_step(self, batch: dict, mode: str) -> None:
        """Perform a validation or test step

        Args:
            batch (torch.Tensor): input batch of data
            mode (str): val or test mode
        """
        new_seq_flag = bool(batch["new_sequence"])
        end_seq_flag = bool(batch["is_last_frame_in_seq"])

        detections = batch["detections"].flatten(0, 1)
        frame_id = batch["frame_id"][0]
        frame_height = batch["frame_height"][0]
        frame_width = batch["frame_width"][0]
        app_vectors = batch["appearance_vectors"].flatten(0, 1)

        if new_seq_flag:
            if mode == "val":
                seq_MOTC_gt = batch["MOTC_gt_file"][0]
                self.results_file = self.evaluator.register_file(seq_MOTC_gt)
            elif mode == "test":
                seq_name = batch["seq"][0]
                self.results_file = self.inferencer.register_sequence(seq_name)
            else:
                raise AssertionError
            self.val_res_buffer = list()
            self._new_sequence()

        # MODEL STEP
        ecc_transform = batch["ecc"][0].cpu().numpy() if "ecc" in batch else None
        tgt_motion_model_preds = self.manager.motion_predictions(
            ecc_transform=ecc_transform
        )

        self.manager.step(
            detections, app_vectors, computed_motion_predictions=tgt_motion_model_preds
        )

        tgt_state = self.manager.current_state
        #  Convert back to image-coords
        normalizer = torch.tensor(
            [frame_width, frame_height, frame_width, frame_height],
            dtype=torch.float32,
            device=self.device,
        )

        for id in tgt_state:
            tgt_state[id]["bbox"] *= normalizer

        for row in convert_MOTC_format(frame_id, tgt_state):
            self.val_res_buffer.append(row)

        if end_seq_flag:
            # Flush buffer
            assert isinstance(self.results_file, str)
            # Truncate non-hit trails of trajectories
            self.val_res_buffer = truncate_tracklets_MOTC_format(self.val_res_buffer)
            with open(self.results_file, "w") as f:
                f.writelines(self.val_res_buffer)

    def _test_or_val_cleanup_epoch_end(self) -> None:
        """Cleanup after each epoch of testing or validation"""
        assert isinstance(self.results_file, str)
        if not os.path.exists(self.results_file) and len(self.val_res_buffer) > 0:
            self.val_res_buffer = truncate_tracklets_MOTC_format(self.val_res_buffer)
            with open(self.results_file, "w") as f:
                f.writelines(self.val_res_buffer)
            self.val_res_buffer = list()

        self.allow_reflection = self._old_allow_reflection

    def validation_step(self, batch, batch_idx) -> None:
        """Perform a validation step

        Args:
            batch (torch.Tensor): input batch of data
            batch_idx (int): batch index
        """
        self._test_or_val_step(batch, mode="val")

    def on_validation_epoch_start(self) -> None:
        """Actions to be performed at the beginning of each validation epoch"""
        self._old_allow_reflection = self.allow_reflection
        self.allow_reflection = False

        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        """Actions to be performed at the end of each validation epoch"""
        self._test_or_val_cleanup_epoch_end()
        detailed_report = self.evaluator.eval()
        self.log_dict(detailed_report, on_epoch=True, on_step=False, batch_size=1)

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """Perform a testing step

        Args:
            batch (torch.Tensor): input batch of data
            batch_idx (int): batch index
        """
        self._test_or_val_step(batch, mode="test")

    def on_test_epoch_start(self) -> None:
        """Actions to be performed at the beginning of each testing epoch"""
        self._old_allow_reflection = self.allow_reflection
        self.allow_reflection = False
        self.inferencer.reset()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        """Actions to be performed at the end of each testing epoch"""
        self._test_or_val_cleanup_epoch_end()

        return super().on_test_epoch_end()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for training"""

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_schedulers = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        }

        lr_scheduler = lr_schedulers[self.lr_scheduler_params.pop("type")](
            optimizer, **self.lr_scheduler_params
        )

        return [optimizer], [lr_scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Callback for checkpoint saving"""
        checkpoint["tracker_params"] = self.manager.tracker.init_hyperparameters
        checkpoint["manager_params"] = self.manager.init_hyperparameters
        checkpoint["training_params"] = self.init_hyperparameters
        checkpoint["motion_params"] = {"type": tracklets._MOTION_MODEL.type()}
        checkpoint["kill_params"] = {
            "min_t": tracklets._MIN_KILL_THRESHOLD,
            "max_t": tracklets._MAX_KILL_THRESHOLD,
            "max_hits": tracklets._MAX_HITS,
        }

    @classmethod
    def init_from_ckpt(cls, ckpt: dict):
        """Load and initialize a model from checkpoint

        Args:
            ckpt (dict): Checkpoint dict with the following keys:
                ("tracker_params", "manager_params", "motion_params", "state_dict", "training_params")
        """
        tracker = TADN(**ckpt["tracker_params"])
        manager = OnlineManager(tracker, **ckpt["manager_params"])
        obj = cls(manager, **ckpt["training_params"])
        obj.load_state_dict(ckpt["state_dict"])
        obj._new_sequence()

        tracklets.set_motion_model(ckpt["motion_params"]["type"])
        tracklets.set_kill_thresholds(**ckpt["kill_params"])

        return obj


def init_model_from_config(cfg: DictConfig) -> OnlineTraining:
    """Initialize model, manager, tracker and components from given configuration.

    Args:
        cfg (ExperimentConfig): Experiment configuration

    Returns:
        OnlineTraining: Model to train
    """
    tracker: TADN = get_tracker(cfg.tracker)
    manager = OnlineManager(
        tracker=tracker, choice_assignment_params=cfg.manager.choice_assignment_params
    )
    model = OnlineTraining(
        manager=manager,
        benchmark=get_evaluation_benchmark(cfg.dataset.type),
        tgt2det_min_threshold=cfg.model_training.tgt2det_min_threshold,
        null_target_weight=cfg.model_training.null_target_weight,
        learning_rate=cfg.model_training.learning_rate,
        allow_reflection=cfg.model_training.allow_reflection,
        lr_scheduler_params=cfg.model_training.lr_scheduler_params,
        assignment_threshold=cfg.model_training.assignment_threshold,
    )
    tracklets.set_motion_model(cfg.tracklets.motion_model)
    tracklets.set_kill_thresholds(
        min_t=cfg.tracklets.min_kill_threshold,
        max_t=cfg.tracklets.max_kill_threshold,
        max_hits=cfg.tracklets.max_kill_threshold_hits,
    )
    metrics.set_metric(cfg.model_training.assignment_metric)
    return model


@hydra.main(version_base=None, config_path="../conf", config_name="online_training")
def main(cfg: DictConfig):
    """Main function

    Args:
        args (Namespace): Config arguments from command line
    """

    model = init_model_from_config(cfg)
    print(model)

    # assert isinstance(cfg.dataset, MOTDatasetConfig)
    # train_dloader, val_dloader = cfg.dataset.build_dataloaders(
    #     batch_size=1, shuffle=False
    # )

    trainer: pl.Trainer = get_trainer(cfg.trainer)

    # trainer.fit(model, train_dataloaders=train_dloader, val_dataloaders=val_dloader)


# Main entry-point
if __name__ == "__main__":
    main()

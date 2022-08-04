"""Script to validate Training Strategy using a LAM-based hypothetical tracker.
Used to estimate a level of performance expected using the Online Training strategy
"""
from argparse import ArgumentParser
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from config4ml.lightning.extra import ConsoleLogger

from ..components import tracklets
from ..components.transformer import TADN
from ..config.data import MOTDatasetConfig
from ..config.experiment import ExperimentConfig
from ..mot import metrics
from ..online_training import OnlineManager, OnlineTraining
from ..utils.bbox import convert_MOTC_format
from ..utils.tracklets import truncate_tracklets_MOTC_format


class Validator(OnlineTraining):
    """Convenience class to support validation/evaluation

    Inherits from:
        OnlineTraining
    """

    def __init__(
        self,
        manager: OnlineManager,
        tgt2det_min_threshold: float = 0.3,
        null_target_weight: int = 1,
        learning_rate: float = 0.0001,
        allow_reflection: bool = True,
        lr_scheduler_params: Dict[str, Any] = ...,
        assignment_threshold: float = 0.1,
        benchmark="MOT17",
    ):
        super().__init__(
            manager=manager,
            tgt2det_min_threshold=tgt2det_min_threshold,
            null_target_weight=null_target_weight,
            learning_rate=learning_rate,
            allow_reflection=allow_reflection,
            lr_scheduler_params=lr_scheduler_params,
            assignment_threshold=assignment_threshold,
            benchmark=benchmark,
        )

        self.manager.lam_validation = True

    def validation_step(self, batch, batch_idx) -> None:
        """Modify validation step to compute LAM"""

        new_seq_flag = bool(batch["new_sequence"])
        end_seq_flag = bool(batch["is_last_frame_in_seq"])
        seq_MOTC_gt = batch["MOTC_gt_file"][0]
        detections = batch["detections"].flatten(0, 1)
        frame_id = batch["frame_id"][0]
        frame_height = batch["frame_height"][0]
        frame_width = batch["frame_width"][0]
        app_vectors = batch["appearance_vectors"].flatten(0, 1)

        if new_seq_flag:
            self.results_file = self.evaluator.register_file(seq_MOTC_gt)
            self.val_res_buffer = list()
            self._new_sequence()

        detections = batch["detections"].flatten(0, 1)
        gt_tgt_locs = batch["gt"].flatten(0, 1)
        gt_tgt_locs_prev = batch["gt_prev"].flatten(0, 1)

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

        self.manager.register_label_assignment_matrix(
            label_am, epoch=self.current_epoch
        )

        # MODEL STEP
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
            self.val_res_buffer = truncate_tracklets_MOTC_format(self.val_res_buffer)
            with open(self.results_file, "w") as f:
                f.writelines(self.val_res_buffer)


def init_model_from_config(cfg: ExperimentConfig) -> OnlineTraining:
    """Initialize a Validator instance from config"""
    tracker: TADN = cfg.tracker.get_tracker()
    manager = OnlineManager(tracker=tracker, **cfg.manager.dict())
    assert isinstance(cfg.dataset, MOTDatasetConfig)
    model = Validator(
        manager=manager,
        benchmark=cfg.dataset.evaluation_benchmark,
        **cfg.model_training.kwargs
    )
    tracklets.set_motion_model(cfg.tracklets.motion_model)
    tracklets.set_kill_thresholds(**cfg.tracklets.kill_threshold_opts)
    metrics.set_metric(cfg.model_training.assignment_metric)

    return model


def main(args):
    """Main script function"""

    cfg = ExperimentConfig.parse_file(args.json_config)

    model = init_model_from_config(cfg)

    assert isinstance(cfg.dataset, MOTDatasetConfig)
    _, val_dloader = cfg.dataset.build_dataloaders(batch_size=1)

    callbacks_list = [cb.callback for cb in cfg.trainer.callbacks]
    logger = ConsoleLogger()

    kwargs = cfg.trainer.dict()
    kwargs["callbacks"] = callbacks_list
    kwargs["logger"] = logger
    trainer = pl.Trainer(**kwargs)

    trainer.validate(model, dataloaders=val_dloader)


# Main entry point
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_config", help="Path to json config file")

    args = parser.parse_args()
    main(args)

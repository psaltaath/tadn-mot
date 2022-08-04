from typing import Optional

from config4ml.lightning import TrainerConfig
from pydantic import BaseModel, validator

from ..components.transformer import TADN

# Local imports
from ._sub_configs import (
    ManagerChoiceAssignmentConfig,
    TrackerDualTransformerConfig,
    TrackerEmbeddingConfig,
    TrackerNullTargetConfig,
    TrackerTransformerConfig,
)
from .data import MOTDatasetConfig, select_dataset
from .utils import json_dumps_for_callables


class TrackerConfig(BaseModel):
    """Configuration for TADN tracking module"""

    transformer_params: TrackerTransformerConfig
    embedding_params: TrackerEmbeddingConfig = TrackerEmbeddingConfig()
    null_target_params: TrackerNullTargetConfig = TrackerNullTargetConfig()
    normalize_transformer_outputs: bool = False

    @validator("transformer_params", pre=True)
    def validate_transformer(cls, v):
        if isinstance(v, TrackerTransformerConfig):
            return v

        assert isinstance(v, dict)
        assert "type" in v

        if v["type"] == "single":
            v = TrackerTransformerConfig.parse_obj(v)
        elif v["type"] == "dual":
            v = TrackerDualTransformerConfig.parse_obj(v)
        else:
            raise AssertionError("Invalid transformer type")
        return v

    def get_tracker(self):
        """Initialize a TADN module given config options"""
        d_model = (
            self.embedding_params.app_embedding_dim
            + self.embedding_params.spatial_embedding_dim
        )
        transformer_model = self.transformer_params.get_transformer(d_model)
        tracker_model = TADN(
            transformer_model=transformer_model,
            embedding_params=self.embedding_params.dict(),
            null_target_params=self.null_target_params.dict(),
            normalize_transformer_outputs=self.normalize_transformer_outputs,
        )
        return tracker_model


class ManagerConfig(BaseModel):
    """Configuration for MOTManager"""

    choice_assignment_params: ManagerChoiceAssignmentConfig = (
        ManagerChoiceAssignmentConfig()
    )


class TrackletsConfig(BaseModel):
    """Configuration for tracklet behavior"""

    motion_model: str = "kalman"
    min_kill_threshold: int = 3
    max_kill_threshold: int = 30
    max_kill_threshold_hits: int = 100

    @property
    def kill_threshold_opts(self):
        """Returns options for determining kill threshold"""
        return {
            "min_t": self.min_kill_threshold,
            "max_t": self.max_kill_threshold,
            "max_hits": self.max_kill_threshold_hits,
        }


class ModelTrainingConfig(BaseModel):
    """Configuration for training parameters"""

    tgt2det_min_threshold: float = 0.3
    null_target_weight: int = 5
    learning_rate: float = 0.0001
    allow_reflection: bool = True
    lr_scheduler_params: dict = {"type": "StepLR", "step_size": 80, "gamma": 0.1}
    assignment_threshold: float = 0.1
    assignment_metric: str = "iou"

    @property
    def kwargs(self):
        kwargs = self.dict()
        del kwargs["assignment_metric"]
        return kwargs


class ExperimentConfig(BaseModel):
    """Global configuration for a single experiment"""
    dataset: Optional[MOTDatasetConfig]
    tracker: TrackerConfig = TrackerConfig(
        transformer_params=TrackerTransformerConfig()
    )
    manager: ManagerConfig = ManagerConfig()
    tracklets: TrackletsConfig = TrackletsConfig()
    model_training: ModelTrainingConfig = ModelTrainingConfig()
    trainer: TrainerConfig = TrainerConfig(logger={"type": "tensorboard", "save_dir": "./tb_logs"})  # type: ignore

    class Config:
        json_dumps = json_dumps_for_callables

    @validator("dataset", pre=True)
    def select_dataset_validator(v) -> MOTDatasetConfig:
        if isinstance(v, MOTDatasetConfig):
            return v
        assert isinstance(v, dict)
        assert "type" in v

        return select_dataset(v)

    def __str__(self) -> str:
        return self.json(indent=2)


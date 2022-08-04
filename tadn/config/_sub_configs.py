from typing import Optional
from pydantic import BaseModel
from ..components.transformer import SingleBranchTransformer, DualBranchTransformer


class TrackerTransformerConfig(BaseModel):
    """Configuration for TADN transformer single-branch architecture"""

    type: str = "single"
    nhead: int = 4
    encoder_num_layers: int = 3
    decoder_num_layers: int = 3

    def get_transformer(self, d_model):
        """Initializes a single-branch TADN Transformer instance"""
        params_dict = self.dict()
        del params_dict["type"]
        return SingleBranchTransformer(d_model, **params_dict)


class TrackerDualTransformerConfig(TrackerTransformerConfig):
    """Configuration for TADN transformer dual-branch architecture"""

    type = "dual"

    def get_transformer(self, d_model):
        """Initializes a dual-branch TADN Transformer instance"""
        params_dict = self.dict()
        del params_dict["type"]
        return DualBranchTransformer(d_model, **params_dict)


class TrackerEmbeddingConfig(BaseModel):
    """Configuration for TADN embedding parameters"""

    dim_multiplier: int = 2
    app_dim: int = 512
    app_embedding_dim: int = 512
    spatial_embedding_dim: int = 512
    spatial_memory_mask_weight: Optional[float] = None


class TrackerNullTargetConfig(BaseModel):
    """Configuration for TADN null-target options"""

    null_target_idx: int = -1


class ManagerChoiceAssignmentConfig(BaseModel):
    """Configuration for Choice Assignment parameters"""

    starting_epoch: int = 5
    ending_epoch: int = 40

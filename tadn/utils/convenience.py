from omegaconf import DictConfig

from ..components.transformer import (
    TADN,
    SingleBranchTransformer,
    DualBranchTransformer,
)


def get_tracker(tracker_cfg: DictConfig):
    """Initialize a TADN module given config options"""
    d_model = (
        tracker_cfg.embedding_params.app_embedding_dim
        + tracker_cfg.embedding_params.spatial_embedding_dim
    )
    transformer_model = get_transformer(
        d_model=d_model, transformer_model_cfg=tracker_cfg.transformer_params
    )
    tracker_model = TADN(
        transformer_model=transformer_model,
        embedding_params=tracker_cfg.embedding_params,
        null_target_params=tracker_cfg.null_target_params,
        normalize_transformer_outputs=tracker_cfg.normalize_transformer_outputs,
    )
    return tracker_model


def get_transformer(d_model, transformer_model_cfg: DictConfig):
    __base_cls__ = None
    if transformer_model_cfg.type == "single":
        __base_cls__ = SingleBranchTransformer
    else:
        __base_cls__ = DualBranchTransformer

    return __base_cls__(
        d_model=d_model,
        nhead=transformer_model_cfg.nhead,
        encoder_num_layers=transformer_model_cfg.encoder_num_layers,
        decoder_num_layers=transformer_model_cfg.decoder_num_layers,
    )


def get_evaluation_benchmark(dataset_type: str) -> str:
    if "MOT" in dataset_type.upper():
        return "MOT17"
    elif "DETRAC" in dataset_type.upper():
        return "DETRAC"
    raise AssertionError("Invalid dataset type for evaluation benchmark")


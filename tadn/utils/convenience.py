from omegaconf import DictConfig
from ..components.transformer import (
    TADN,
    SingleBranchTransformer,
    DualBranchTransformer,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def parse_callbacks(callback_cfgs: list) -> list:
    cbs = []
    for cb in callback_cfgs:
        if cb.type == "model_checkpoint":
            opts = dict(**cb)
            del opts["type"]
            cbs.append(ModelCheckpoint(**opts))
        else:
            raise NotImplementedError
    return cbs

def parse_logger(logger_cfg: DictConfig):
    if logger_cfg.type == "tensorboard":
        return TensorBoardLogger(save_dir=logger_cfg.save_dir)
    else:
        raise NotImplementedError

def get_trainer(trainer_cfg: DictConfig) -> pl.Trainer:

    callbacks = parse_callbacks(trainer_cfg.callbacks)
    logger = parse_logger(trainer_cfg.logger)

    trainer = pl.Trainer(
        accelerator=trainer_cfg.accelerator,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        max_epochs=trainer_cfg.max_epochs,
        check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger
    )

    return trainer



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

    


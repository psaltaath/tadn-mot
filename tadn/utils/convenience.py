from omegaconf import DictConfig
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
        return TensorBoardLogger()
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
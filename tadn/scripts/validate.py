"""Script to perform evaluation using a pretrained model"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from config4ml.lightning.extra import ConsoleLogger

from ..config.data import MOTDatasetConfig
from ..config.experiment import ExperimentConfig
from .inference import load_from_ckpt


def main(args):
    """Main script function"""
    model = load_from_ckpt(args.ckpt, args.json_config)
    model.manager.lam_validation = False

    cfg = ExperimentConfig.parse_file(args.json_config)

    assert isinstance(cfg.dataset, MOTDatasetConfig)
    _, val_dloader = cfg.dataset.build_dataloaders(batch_size=1)

    callbacks_list = [cb.callback for cb in cfg.trainer.callbacks]
    logger = ConsoleLogger()

    kwargs = cfg.trainer.dict()
    kwargs["callbacks"] = callbacks_list
    kwargs["logger"] = logger
    trainer = pl.Trainer(**kwargs)

    trainer.validate(model, dataloaders=val_dloader)


# Main entry-point
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt", type=str, help="Path to checkpoint")
    parser.add_argument(
        "json_config", type=str, help="Path to json config used for training"
    )

    args = parser.parse_args()
    main(args)

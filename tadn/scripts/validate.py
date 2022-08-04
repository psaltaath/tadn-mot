from argparse import ArgumentParser
from .validate_LAM import Validator, init_model_from_config
from ..config.experiment import ExperimentConfig
from ..config.data import MOTDatasetConfig
from config4ml.lightning.extra import ConsoleLogger
import pytorch_lightning as pl
import torch


def main(args):

    cfg = ExperimentConfig.parse_file(args.json_config)

    model = init_model_from_config(cfg)

    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.manager.lam_validation = False

    assert isinstance(cfg.dataset, MOTDatasetConfig)
    _, val_dloader = cfg.dataset.build_dataloaders(batch_size=1)

    callbacks_list = [cb.callback for cb in cfg.trainer.callbacks]
    logger = ConsoleLogger()

    kwargs = cfg.trainer.dict()
    kwargs["callbacks"] = callbacks_list
    kwargs["logger"] = logger
    trainer = pl.Trainer(**kwargs)

    trainer.validate(model, dataloaders=val_dloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--json_config")

    args = parser.parse_args()
    main(args)

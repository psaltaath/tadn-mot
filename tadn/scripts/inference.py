"""Script to perform inference using a pretrained model"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from ..config.data import MOTDatasetConfig
from ..config.experiment import ExperimentConfig
from ..online_training import init_model_from_config


def load_from_ckpt(ckpt_file, cfg_file):
    """Load model from checkpoint"""
    cfg = ExperimentConfig.parse_file(cfg_file)

    model = init_model_from_config(cfg)

    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["state_dict"])

    return model


def main(args):
    """Main script function"""
    model = load_from_ckpt(args.ckpt, args.json_config)
    print(model)

    cfg = ExperimentConfig.parse_file(args.json_config)
    assert isinstance(cfg.dataset, MOTDatasetConfig)
    cfg.dataset.skip_first_frame = False
    train_dloader, val_dloader = cfg.dataset.build_dataloaders()

    dloaders = [val_dloader]
    if args.inference_train:
        dloaders.append(train_dloader)

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=dloaders)


# Main entry-point
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt", type=str, help="Path to checkpoint")
    parser.add_argument(
        "json_config", type=str, help="Path to json config used for training"
    )
    parser.add_argument(
        "--inference_train", action="store_true", help="Inference also on training set"
    )

    args = parser.parse_args()
    main(args)

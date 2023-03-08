"""Script to perform inference using a pretrained model"""
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch
import hydra

from ..data.utils import build_datasets, build_dataloaders
from ..online_training import init_model_from_config


def load_from_ckpt(cfg: DictConfig):
    """Load model from checkpoint"""

    model = init_model_from_config(cfg)

    ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    return model


@hydra.main(version_base=None, config_path="../../conf", config_name="inference")
def main(cfg: DictConfig):
    """Main script function"""
    model = load_from_ckpt(cfg)
    print(model)

    cfg.dataset.skip_first_frame = False
    val_dset = build_datasets(dataset_cfg=cfg.dataset, skip_train=True)
    val_dloader = build_dataloaders(
        val_dset, dataloader_cfg=cfg.dataset.dataloader
    )

    print(len(val_dset))

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=val_dloader)


# Main entry-point
if __name__ == "__main__":
    main()

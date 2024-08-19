import torch
import pytorch_lightning as pl
import rootutils
import hydra
import datetime
from omegaconf import DictConfig

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOTPATH / "configs")

import src.utils as utils

@hydra.main(config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig):
    test_func = lambda x: torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1] ** 2 + torch.tan(x[:, 0]))
    sample_ranges = [[-1, 1], [-1, 1]]
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data_module, func=test_func, range=sample_ranges)

    model: pl.LightningModule = hydra.utils.instantiate(cfg.model.model)
    model.configure_optimizers(cfg.model.optimizer)

    loss_logger = utils.LossLogger()
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[loss_logger])

    print("Starting training")
    trainer.fit(model, data_module)
    
    print("Starting testing")
    
    trainer.test(model, datamodule=data_module)

    loss_logger.plot_losses()
    torch.save(model.state_dict(), f"{ROOTPATH}\ckpt\MLP{datetime.datetime.now().strftime('_%d_%m_%Y_%H_%M_%S')}.pth")
    return 

if __name__ == '__main__':
    main()
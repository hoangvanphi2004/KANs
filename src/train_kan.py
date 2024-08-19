import torch
import math
import torch
import pytorch_lightning as pl
import rootutils
import hydra
from omegaconf import DictConfig
import datetime

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOTPATH / "configs")

import src.utils as utils

@hydra.main(config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig):
    # Set up the data module
    test_func = lambda x: torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1] ** 2 + torch.tan(x[:, 0]))
    sample_ranges = [[-1, 1], [-1, 1]]
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data_module, func=test_func, range=sample_ranges)

    data_module.setup(stage='fit')
    # Set up the model
    grid_finer = cfg.grid_sizes
    old_model = None
    new_model = None
    learning_rate = cfg.model.optimizer.lr

    # Set up the trainer
    loss_logger = utils.LossLogger()

    # Train and test the model
    for grid in grid_finer:
        new_model: pl.LightningModule = hydra.utils.instantiate(cfg.model.model)
        if(old_model != None):
            new_model.initial_grid_from_other_model(old_model, x=data_module.train.x)
        # new_model.configure_optimizers(cfg.model.optimizer)
        new_model.configure_loss_func(**cfg.model.loss)
        
        trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[loss_logger])
        print(f"Starting training with grid size {grid}")
        trainer.fit(new_model, data_module)

        print(f"Starting testing with grid size {grid}")
        trainer.test(new_model, datamodule=data_module)
        old_model = new_model
        learning_rate /= (3 + math.log10(grid) / 2) 
    
    new_model.plot()
    loss_logger.plot_losses()
    
    torch.save(new_model.state_dict(), f"{ROOTPATH}\ckpt\KAN{datetime.datetime.now().strftime('_%d_%m_%Y_%H_%M_%S')}.pth")
    
    return 

if __name__ == '__main__':
    main()


import pytorch_lightning as pl
import matplotlib.pyplot as plt

class LossLogger(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        print('e')
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        self.val_losses.append(val_loss.item())

    def plot_losses(self):
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot training and validation loss
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.set_title('Training and Validation Loss Over Epochs')

            plt.tight_layout()
            plt.show()
        else:
            print("No data to plot.")
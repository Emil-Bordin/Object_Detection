import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
import torch

from Data_Loading import DataLoading
from Model_Initialization import FasterRCNN

if __name__ == "__main__":
    model = FasterRCNN(8)

    data_loading = DataLoading()

    train_loader, val_loader, test_loader = data_loading.load_data()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(devices=1, accelerator="auto", max_epochs=100, default_root_dir="/hands_on", logger=tb_logger, log_every_n_steps=10,
                         callbacks=EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, verbose=True, mode='max'))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    torch.save(model, r"/home/jovyan/work/FasterRCNN/FasterRCNN.pt") #Ich habe auf dem Server einen neuen Ordner mit dem Namen "FasterRCNN" erstellt, deshalb lautet der Pfad so, damit das CNN im Ordner abgespeichert wird

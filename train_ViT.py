# train.py

import torch
from pytorch_lightning import Trainer
from dataloader.dataloader import DLWCDataModule
from ViT import WeatherViT

if __name__ == "__main__":
    vars = [
        # list your 35 channel names in order
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]

    dm = DLWCDataModule(
        root_dir="data",
        variables=vars,
        batch_size=8,
        test_batch_size=8,
        #num_workers=4,
        list_train_intervals=[(0, 1000), (1000, 2000), (2000, 3000)],  # Example intervals
    )
    dm.setup()

    model = WeatherViT(
        in_channels=len(vars),  # 35
        out_channels=len(vars),  # 35
        lr=3e-4,
    )

    trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=50,
    )
    trainer.fit(model, dm)

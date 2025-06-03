import pytorch_lightning as pl
from model import SimpleWeatherCNN
from dataloader.dataloader import DLWCDataModule

if __name__ == "__main__":
    root_dir = "data"
    variables = [
        'z_20000', 'z_30000', 'z_50000', 'z_70000', 'z_85000', 'z_92500', 'z_100000',
        't_20000', 't_30000', 't_50000', 't_70000', 't_85000', 't_92500', 't_100000',
        'u_20000', 'u_30000', 'u_50000', 'u_70000', 'u_85000', 'u_92500', 'u_100000',
        'v_20000', 'v_30000', 'v_50000', 'v_70000', 'v_85000', 'v_92500', 'v_100000',
        'r_20000', 'r_30000', 'r_50000', 'r_70000', 'r_85000', 'r_92500', 'r_100000'
    ]
    in_channels = len(variables)
    out_channels = len(variables)

    dm = DLWCDataModule(
        root_dir=root_dir,
        variables=variables,
        list_train_intervals=[3],
        batch_size=2,
        test_batch_size=2,
    )
    #dm.setup()

    model = SimpleWeatherCNN(in_channels=in_channels, out_channels=out_channels)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)